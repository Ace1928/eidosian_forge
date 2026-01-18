import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
class AsyncCheckpointHelper:
    """Helper class for async checkpoint."""

    def __init__(self, checkpointer_impl, root=None, **kwargs):
        """Initialize AsyncCheckpoint.

    Args:
      checkpointer_impl: The Checkpoint class to power the AsyncCheckpoint.
      root: The root object to checkpoint. `root` may be a trackable object or
        `WeakRef` of a trackable object.
      **kwargs: The keyword arguments representing the checkpointed variables.

    Raises:
      AttributeError: when checkpointer_impl is None.
    """
        if root:
            trackable_root = root() if isinstance(root, weakref.ref) else root
            kwargs['root'] = trackable_root
            trackable_root._maybe_initialize_trackable()
        if checkpointer_impl is None:
            raise AttributeError('checkpointer_impl cannot be None for AsyncCheckpointHelper.')
        self._checkpointer_impl = checkpointer_impl
        self._checkpoint_items = kwargs
        self._checkpoint = None
        self.checkpointer()
        self._checkpoint_options = None
        self._initialized = False
        self._async_write_done_callback = None
        self._original_nodes = None
        self._object_map = None
        self._tpu_embedding_objects = None
        self._default_device = device_util.current() or 'CPU:0'
        self._default_device = device_util.canonicalize(self._default_device)
        self._save_file_prefix = None
        self._use_checkpoint_save = False
        self._async_save_thread = None
        self._queue = queue.Queue(maxsize=1)
        atexit.register(self._join_async_save_thread)
        self._async_error = None
        global _END_TIME_OF_LAST_ASYNC_WRITE
        with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
            if _END_TIME_OF_LAST_ASYNC_WRITE is None:
                _END_TIME_OF_LAST_ASYNC_WRITE = time.time()

    @def_function.function
    def _copy_from_cpu(self):
        """Copy the checkpointed variables from the host CPU to the accelerator.

    TODO(chienchunh): Get the concrete function before firstly called to avoid
                      hangining the accelerators idle during function tracing.
    """
        for accelerator_var, cpu_var in self._object_map.items():
            if isinstance(accelerator_var, ShardedVariable) or hasattr(accelerator_var, _TPU_EMBEDDING_ATTR):
                continue
            with ops.device(accelerator_var.device):
                accelerator_var.assign(cpu_var.read_value())

    @def_function.function
    def _copy_to_cpu(self):
        """Copy the checkpointed variables from the accelerator to the host CPU.

    TODO(chienchunh): Get the concrete function before firstly called to avoid
                      hangining the accelerators idle during function tracing.
    """
        for accelerator_var, cpu_var in self._object_map.items():
            if isinstance(accelerator_var, ShardedVariable) or hasattr(accelerator_var, _TPU_EMBEDDING_ATTR):
                continue
            with ops.device(cpu_var.device):
                cpu_var.assign(accelerator_var.read_value())
        for tpu_embedding in self._tpu_embedding_objects:
            tpu_embedding._retrieve_variables()

    def _traverse_variables(self, to_traverse, visited):
        """Create the copied nodes and variables while traversing the nodes.

    This method performs a BFS to traverse the nodes while avoiding duplicated
    visits. Throughout the process, self._mapping, self._original_nodes, and
    self._var_pairs are populated.

    Args:
      to_traverse: A deque that stores the nodes to be traversed.
      visited: A list of nodes that have been visited.
    """
        while to_traverse:
            current_trackable = to_traverse.popleft()
            self._original_nodes.append(current_trackable)
            if isinstance(current_trackable, (Variable, ShardedVariable)):
                self._copy_trackable(current_trackable)
            if hasattr(current_trackable, _TPU_EMBEDDING_ATTR):
                self._handle_tpu_embedding(current_trackable)
            for child in current_trackable._trackable_children(save_type='checkpoint').values():
                if child in visited:
                    continue
                visited.add(child)
                to_traverse.append(child)

    def checkpointer(self):
        """Gets or creates the underlying Checkpoint instance."""
        if self._checkpoint is None:
            self._checkpoint = self._checkpointer_impl(**self._checkpoint_items)
        return self._checkpoint

    def _ensure_initialized(self):
        """Initialize the async checkpoint internal state."""
        if self._initialized:
            return
        self._original_nodes = []
        self._object_map = object_identity.ObjectIdentityDictionary()
        self._tpu_embedding_objects = []
        to_traverse = collections.deque([])
        visited = object_identity.ObjectIdentitySet()
        for v in self._checkpoint_items.values():
            if isinstance(v, (Variable, ShardedVariable)):
                self._copy_trackable(v)
            elif hasattr(v, _TPU_EMBEDDING_ATTR):
                self._handle_tpu_embedding(v)
            to_traverse.append(v)
            visited.add(v)
        self._traverse_variables(to_traverse, visited)
        for current_trackable in self._original_nodes:
            if 'get_slot_names' in dir(current_trackable):
                slot_names = current_trackable.get_slot_names()
                for slot_name in slot_names:
                    for original_variable in self._original_nodes:
                        if not isinstance(original_variable, Variable):
                            continue
                        try:
                            original_slot_variable = current_trackable.get_slot(original_variable, slot_name)
                        except (AttributeError, KeyError):
                            continue
                        if isinstance(original_slot_variable, (Variable, ShardedVariable)):
                            self._copy_trackable(original_slot_variable)
        save_counter = self.checkpointer().save_counter.numpy()
        logging.info("Initializing async checkpoint's save_counter: %d", save_counter)
        self.checkpointer()._saver._object_map = self._object_map
        self._async_save_thread = threading.Thread(target=self._async_save, daemon=True)
        self._async_save_thread.start()
        self._initialized = True

    def _check_async_thread_error(self):
        """Expose the most recent error from the async saving thread to the caller.
    """
        if self._async_error:
            e = self._async_error
            self._async_error = None
            logging.error('Propagating the most recent error from the async thread before joining: %s', str(e))
            raise e

    def _join_async_save_thread(self):
        """Join the async save thread.

    The steps for terminating the async save thread:
    1). Put will succeed when the last async save event is done. Putting a false
        triggers the async save thread's while loop to end. We use put instead
        of sync because sync does not have a timeout argument.
    2). Join the async save thread. (The thread may finish before joining.)
    """
        try:
            self._queue.put(False, timeout=300)
            logging.info('Joining the async save thread.')
            if self._async_save_thread is not None:
                self._async_save_thread.join()
        except queue.Full:
            logging.error('Timeout waiting for the async save thread; terminating the thread instead. The last checkpoint may be incomeplete.')
        finally:
            self._check_async_thread_error()

    def _async_save(self):
        """The thread function for the async checkpoint save."""
        with context.executor_scope(executor.new_executor(enable_async=False, enable_streaming_enqueue=False)):
            while self._queue.get():
                logging.info('Starting async checkpoint save on the device: %s', self._default_device)
                async_save_start_time = time.time()
                try:
                    with ops.device(self._default_device):
                        with checkpoint_context.async_metrics_context():
                            if self._use_checkpoint_save:
                                self.checkpointer().save(self._save_file_prefix, self._checkpoint_options)
                            else:
                                self.checkpointer()._write(self._save_file_prefix, options=self._checkpoint_options, write_done_callback=self._async_write_done_callback)
                except Exception as e:
                    self._async_error = e
                finally:
                    self._queue.task_done()
                async_save_end_time = time.time()
                metrics.AddAsyncCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(async_save_start_time, async_save_end_time))
                global _END_TIME_OF_LAST_ASYNC_WRITE
                with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
                    metrics.AddTrainingTimeSaved(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_ASYNC_WRITE, async_save_start_time))
                    _END_TIME_OF_LAST_ASYNC_WRITE = async_save_start_time
        logging.info('Async save thread reached the end of the execution.')

    def _copy_for_variable(self, original_var):
        """Create a new instance for the input trackable.

    Args:
      original_var: Input Variable object to be copied.
    """
        op_device = pydev.DeviceSpec.from_string(original_var.device).replace(device_type='CPU', device_index=0).to_string()
        with ops.device(op_device):
            new_var = UninitializedVariable(trainable=original_var.trainable, shape=original_var.shape, dtype=original_var.dtype, name=original_var._shared_name)
        self._object_map[original_var] = new_var

    def _copy_for_sharded_variable(self, original_var):
        """Create a new instance for the input ShardedVariable.

    Args:
      original_var: Input ShardedVariable object to be copied.
    """
        copied_vars = []
        for v in original_var._variables:
            self._copy_for_variable(v)
            copied_vars.append(self._object_map[v])
        self._object_map[original_var] = ShardedVariable(copied_vars, name=original_var.name)

    def _copy_trackable(self, original_trackable):
        """Create a new instance for the input trackable.

    Args:
      original_trackable: The trackable instance to be copied.

    Raises:
      AttributeError: if the input trackable is not Variable or ShardedVariable.
    """
        if isinstance(original_trackable, ShardedVariable):
            self._copy_for_sharded_variable(original_trackable)
        elif isinstance(original_trackable, Variable):
            self._copy_for_variable(original_trackable)
        else:
            raise AttributeError('Only Variable or ShardedVariable can be copied.')

    def _handle_tpu_embedding(self, tpu_embedding):
        """Handle TPUEmbedding.

    Args:
      tpu_embedding: TPUEmbedding object to be handled.

    Raises:
      AttributeError: if the input trackable is not TPUEmbedding type.
    """
        if not hasattr(tpu_embedding, _TPU_EMBEDDING_ATTR) or not callable(tpu_embedding._create_copy_for_async_checkpoint):
            raise AttributeError('Expecting TPUEmbedding type; got %s' % type(tpu_embedding))
        new_embedding = tpu_embedding._create_copy_for_async_checkpoint(feature_config=tpu_embedding._feature_config, optimizer=tpu_embedding._table_config[0] if tpu_embedding._table_config else None, pipeline_execution_with_tensor_core=tpu_embedding._pipeline_execution_with_tensor_core)
        self._object_map[tpu_embedding] = new_embedding
        if tpu_embedding not in self._tpu_embedding_objects:
            self._tpu_embedding_objects.append(tpu_embedding)

    @property
    def save_counter(self):
        """An integer variable numbering the checkpoint events.

    This is maintained by the underlying tf.train.Checkpoing object employed by
    AsyncCheckpoint class. The number starts at 0 and gets incremented for each
    checkpoint event.

    Returns:
      The save counter variable.
    """
        return self.checkpointer().save_counter

    def write(self, save_path, options=None):
        """Save the checkpointed variables.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.

    Returns:
      The full path of the checkpoint file.
    """
        self._write(save_path, options)

    def _write(self, save_path, options=None, write_done_callback=None):
        """Save the checkpointed variables.

    This method has exactly the same logic as save(), except it does not
    increment the underlying save_counter, which is done by the caller, e.g.,
    CheckpointManager.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.
      write_done_callback: Optional callback function executed after the async
        write is done.

    Returns:
      The full path of the checkpoint file.
    """
        self._ensure_initialized()
        write_start_time = time.time()
        self._queue.join()
        self._copy_to_cpu()
        self._check_async_thread_error()
        context.async_wait()
        self._save_file_prefix = save_path
        self._use_checkpoint_save = False
        self._checkpoint_options = copy.copy(options) if options else None
        if self._checkpoint_options:
            self._checkpoint_options.experimental_enable_async_checkpoint = False
        self._async_write_done_callback = write_done_callback
        self._queue.put(True)
        write_end_time = time.time()
        metrics.AddCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(write_start_time, write_end_time))
        return save_path

    def save(self, save_path, options=None):
        """Save the checkpointed variables.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.

    Returns:
      The full path of the checkpoint file.
    """
        self._ensure_initialized()
        save_start_time = time.time()
        self._queue.join()
        self._copy_to_cpu()
        self._check_async_thread_error()
        save_counter = self.checkpointer().save_counter.numpy() + 1
        full_path = '{}-{}'.format(save_path, save_counter)
        context.async_wait()
        self._save_file_prefix = save_path
        self._use_checkpoint_save = True
        self._checkpoint_options = copy.copy(options) if options else None
        if self._checkpoint_options:
            self._checkpoint_options.experimental_enable_async_checkpoint = False
        self._queue.put(True)
        save_end_time = time.time()
        metrics.AddCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(save_start_time, save_end_time))
        return full_path

    def read(self, save_path, options=None):
        """Restore the checkpointed variables.

    This method has exactly the same logic as restore(). This method is
    implemented only to fulfill the duty of subclassing tf.train.Checkpoint.

    Args:
      save_path: The full name of the checkpoint file to be restored.
      options: CheckpointOption instance.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration. See tf.train.Checkpoint.restore()
      for more details.
    """
        return self.restore(save_path, options)

    def restore(self, save_path, options=None):
        """Restore the checkpointed variables.

    Args:
      save_path: The full name of the checkpoint file to be restored.
      options: CheckpointOption instance.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration. See tf.train.Checkpoint.restore()
      for more details.
    """
        self._checkpoint_options = copy.copy(options) if options else self._checkpoint_options
        if self._checkpoint_options:
            self._checkpoint_options.experimental_enable_async_checkpoint = False
        self._queue.join()
        status = self.checkpointer().restore(save_path, self._checkpoint_options)
        if self._initialized:
            self._copy_from_cpu()
        return status

    def sync(self):
        """Sync on any ongoing save or restore events."""
        self._queue.join()
        logging.info('Sync on ongoing save/restore.')