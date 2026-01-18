import atexit
import os
import re
import socket
import threading
import uuid
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
class _DumpingCallback(object):
    """An object holding the states surrounding the dumping callback."""

    def __init__(self, dump_root, tensor_debug_mode, circular_buffer_size, op_regex, tensor_dtypes):
        self._dump_root = dump_root
        self._tfdbg_run_id = _get_tfdbg_run_id()
        self._tensor_debug_mode = tensor_debug_mode
        self._circular_buffer_size = circular_buffer_size
        self._op_regex = op_regex
        self._tensor_dtypes = tensor_dtypes
        self._hostname = socket.gethostname()
        self._source_file_paths = []
        self._stack_frame_to_id = dict()
        self._context_to_id = dict()
        self._function_to_graph_id = dict()
        self._op_type_to_context_id = dict()
        self._symbolic_tensor_counter = 0
        self._tensor_aliases = dict()
        self._source_file_paths_lock = threading.Lock()
        self._stack_frame_to_id_lock = threading.Lock()
        self._context_lock = threading.Lock()
        self._symbolic_tensor_counter_lock = threading.Lock()
        self._placeholder_to_debug_tensor = object_identity.ObjectIdentityDictionary()
        self._writer = None

    def function_callback(self, function):
        """A callback to be called on creation of ConcreteFunctions."""
        graph_id = self._get_context_id(function.graph)
        with self._context_lock:
            self._function_to_graph_id[function] = graph_id

    @property
    def dump_root(self):
        return self._dump_root

    @dump_root.setter
    def dump_root(self, dump_root):
        if self._dump_root != dump_root:
            self._dump_root = dump_root
            self._writer = None

    @property
    def tfdbg_run_id(self):
        return self._tfdbg_run_id

    @property
    def tensor_debug_mode(self):
        return self._tensor_debug_mode

    @property
    def circular_buffer_size(self):
        return self._circular_buffer_size

    def get_writer(self):
        """Get the debug events writer for the currently configured dump root."""
        if not self._writer:
            self._writer = debug_events_writer.DebugEventsWriter(self._dump_root, self._tfdbg_run_id, circular_buffer_size=self._circular_buffer_size)
        return self._writer

    def _get_context_id(self, context):
        """Get a unique ID for an op-construction context (e.g., a graph).

    If the graph has been encountered before, reuse the same unique ID.
    When encountering a new context (graph), this methods writes a DebugEvent
    proto with the debugged_graph field to the proper DebugEvent file.

    Args:
      context: A context to get the unique ID for. Must be hashable. E.g., a
        Graph object.

    Returns:
      A unique ID for the context.
    """
        if context in self._context_to_id:
            return self._context_to_id[context]
        graph_is_new = False
        with self._context_lock:
            if context not in self._context_to_id:
                graph_is_new = True
                context_id = _get_id()
                self._context_to_id[context] = context_id
        if graph_is_new:
            self.get_writer().WriteDebuggedGraph(debug_event_pb2.DebuggedGraph(graph_id=context_id, graph_name=getattr(context, 'name', None), outer_context_id=self._get_outer_context_id(context)))
        return self._context_to_id[context]

    def _get_outer_context_id(self, graph):
        """Get the ID of the immediate outer context of the input graph.

    Args:
      graph: The graph (context) in question.

    Returns:
      If an outer context exists, the immediate outer context name as a string.
      If such as outer context does not exist (i.e., `graph` is itself
      outermost), `None`.
    """
        if hasattr(graph, 'outer_graph') and graph.outer_graph:
            return self._get_context_id(graph.outer_graph)
        else:
            return None

    def _write_source_file_content(self, file_path):
        """Send the content of a source file via debug-events writer.

    Args:
      file_path: Path to the source file.

    Returns:
      An int index for the file.
    """
        if file_path in self._source_file_paths:
            return self._source_file_paths.index(file_path)
        with self._source_file_paths_lock:
            if file_path not in self._source_file_paths:
                lines = None
                if source_utils.is_extension_uncompiled_python_source(file_path):
                    try:
                        lines, _ = source_utils.load_source(file_path)
                    except IOError as e:
                        logging.warn('Failed to read source code from path: %s. Reason: %s', file_path, e)
                writer = self.get_writer()
                writer.WriteSourceFile(debug_event_pb2.SourceFile(file_path=file_path, host_name=self._hostname, lines=lines))
                self._source_file_paths.append(file_path)
            return self._source_file_paths.index(file_path)

    def _process_stack_frames(self):
        """Process stack frames.

    Send the content of source-files, on a best-effort basis.

    Returns:
      A list of stack frame IDs.
    """
        stack_frames = tf_stack.extract_stack()
        stack_frame_ids = []
        writer = None
        for file_path, lineno, func, _ in stack_frames:
            abs_path = os.path.abspath(file_path)
            if (abs_path, lineno, func) in self._stack_frame_to_id:
                stack_frame_ids.append(self._stack_frame_to_id[abs_path, lineno, func])
                continue
            with self._stack_frame_to_id_lock:
                if (abs_path, lineno, func) not in self._stack_frame_to_id:
                    stack_frame_id = _get_id()
                    self._stack_frame_to_id[abs_path, lineno, func] = stack_frame_id
                    file_index = self._write_source_file_content(abs_path)
                    file_line_col = graph_debug_info_pb2.GraphDebugInfo.FileLineCol(file_index=file_index, line=lineno, func=func)
                    stack_frame_with_id = debug_event_pb2.StackFrameWithId(id=stack_frame_id, file_line_col=file_line_col)
                    writer = self.get_writer()
                    writer.WriteStackFrameWithId(stack_frame_with_id)
                stack_frame_ids.append(self._stack_frame_to_id[abs_path, lineno, func])
        code_location = debug_event_pb2.CodeLocation(host_name=self._hostname, stack_frame_ids=stack_frame_ids)
        return code_location

    def _process_v1_graph_mode_tensor(self, op_type, tensor, debug_tensor, tensor_debug_mode):
        """For V1 graph mode, determine what tensor to output from callback.

    Args:
      op_type: Type of the op that outputs the original symbolic tensor.
      tensor: The original output symbolic tensor.
      debug_tensor: The debugger-instrumented tensor.
      tensor_debug_mode: Debug mode used, a tfdbg TensorDebugMode enum.

    Returns:
      A symbolic tensor to be returned by the dumping op_callback.
    """
        if op_type in ('Placeholder', 'PlaceholderWithDefault'):
            self._placeholder_to_debug_tensor[tensor] = debug_tensor
            return tensor
        elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR and op_type != 'Const':
            self._tensor_aliases[debug_tensor.name] = tensor.name
            return debug_tensor
        else:
            with self._symbolic_tensor_counter_lock:
                identity_name = 'tfdbg_identity_%d' % self._symbolic_tensor_counter
            identity = array_ops.identity(tensor, name=identity_name)
            identity.op._add_control_input(debug_tensor.op)
            self._tensor_aliases[identity.name] = tensor.name
            return identity

    def _instrument_symbolic_tensors(self, tensors, op_type, op_name, tfdbg_context_id, tensor_ids):
        """Add debugging instrumentation for symbolic (i.e., non-eager) tensors.

    The detailed fashion in which the tensors are instrumented is determined
    by the tensor_debug_mode configured for the currently enabled dumping
    callback.

    Args:
      tensors: A tuple of Tensors to instrument. It is assumed that their
        ordering corresponds to the ordering of output tensors of an original
        op. Output slot indices (0-based) will be generated based on the
        ordering.
      op_type: Type name of the op that emits the Tensors (e.g., "MatMul").
      op_name: Name of the op that emits the Tensors (e.g., "dense_1/MatMul").
      tfdbg_context_id: A unique ID for the context that the op belongs to
        (e.g., a graph).
      tensor_ids: A list of unique ID numbers for the tensors, for tfdbg's
        internal use.

    Returns:
      Non-eager Tensors that override the `tensors` as the output of the op
      that originally generated `tensors`. In some cases (e.g., non-V1 graph
      mode), this may be `None`, as the instrumentation can simply rely on
      automatic control dependencies (see `auto_control_deps.py`) instead of
      tensor overriding.
    """
        tensor_debug_mode = self._tensor_debug_mode
        debug_urls = ['file://%s' % self._dump_root]
        is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
        instrumented_tensors = [] if is_v1_graph_mode else None
        for output_slot, tensor in enumerate(tensors):
            with self._symbolic_tensor_counter_lock:
                debug_identity_name = 'DebugIdentityV2_%d' % self._symbolic_tensor_counter
            debug_identity_op_kwargs = {'tfdbg_context_id': tfdbg_context_id, 'op_name': op_name, 'output_slot': output_slot, 'tensor_debug_mode': self._tensor_debug_mode, 'debug_urls': debug_urls, 'name': debug_identity_name, 'circular_buffer_size': self._circular_buffer_size, 'tfdbg_run_id': self._tfdbg_run_id}
            if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
                if not self._should_dump_tensor(op_type, tensor.dtype) or not tensor.dtype.is_numpy_compatible:
                    if is_v1_graph_mode:
                        instrumented_tensors.append(tensor)
                    continue
                if is_v1_graph_mode and (not tensor.dtype.is_numpy_compatible):
                    instrumented_tensors.append(tensor)
                    continue
                debug_tensor = gen_debug_ops.debug_identity_v2(constant_op.constant([], dtype=dtypes.float32), **debug_identity_op_kwargs)
                if is_v1_graph_mode:
                    instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
            elif tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH, debug_event_pb2.TensorDebugMode.SHAPE):
                dtype = tensor.dtype
                dtype_is_dumpable = tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH) and dtype.is_floating or (tensor_debug_mode == debug_event_pb2.TensorDebugMode.SHAPE and (dtype.is_floating or dtype.is_integer or dtype.is_bool))
                if not self._should_dump_tensor(op_type, tensor.dtype) or not dtype_is_dumpable:
                    if is_v1_graph_mode:
                        instrumented_tensors.append(tensor)
                    continue
                debug_tensor = gen_debug_ops.debug_identity_v2(gen_debug_ops.debug_numeric_summary_v2(tensor, tensor_id=tensor_ids[output_slot], tensor_debug_mode=self._tensor_debug_mode, output_dtype=dtypes.float64), **debug_identity_op_kwargs)
                if is_v1_graph_mode:
                    instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
            elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
                if not self._should_dump_tensor(op_type, tensor.dtype) or not tensor.dtype.is_numpy_compatible:
                    if is_v1_graph_mode:
                        instrumented_tensors.append(tensor)
                    continue
                debug_tensor = gen_debug_ops.debug_identity_v2(tensor, **debug_identity_op_kwargs)
                if is_v1_graph_mode:
                    instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
            else:
                raise NotImplementedError('Symbolic tensor instrumentation is not implemented for debug mode %s' % self._tensor_debug_mode)
        return instrumented_tensors

    def _dump_eager_tensors(self, tensors, op_type, input_tensor_ids, output_tensor_device_ids, graph_id=None):
        """Dump the value of eager tensors.

    The destination of the dumping is determined by the dump_root of the
    currently enabled dumping callback. The tensors may be transformed prior to
    dumping (e.g., reduced as summary statistics such as minimum, maximum and
    arithmetic  mean). The details of this transformation (if any) depends on
    the tensor_debug_mode of the currently enabled dumping callback.

    Args:
      tensors: The EagerTensors whose values are to be dumped, with or without
        value transform.
      op_type: Type of the op that generates the tensors, as a string.
      input_tensor_ids: IDs of the input EagerTensors to the op.
      output_tensor_device_ids: Debugged-generated IDs for the devices on which
        the output tensors are allocated, as a `list` of `int`s. Must match
        `tensors` in length.
      graph_id: ID of the executed graph, applicable only to eager execution of
        a FuncGraph.

    Returns:
      A tfdbg Execution protocol buffer.
    """
        tensor_debug_mode = self._tensor_debug_mode
        output_tensor_ids = [t._id for t in tensors]
        assert len(tensors) == len(output_tensor_device_ids)
        if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
            return debug_event_pb2.Execution(op_type=op_type, graph_id=graph_id, num_outputs=len(tensors), input_tensor_ids=input_tensor_ids, output_tensor_ids=output_tensor_ids, output_tensor_device_ids=output_tensor_device_ids, tensor_debug_mode=tensor_debug_mode, code_location=self._process_stack_frames())
        elif tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH, debug_event_pb2.TensorDebugMode.SHAPE, debug_event_pb2.TensorDebugMode.FULL_TENSOR):
            execution_proto = debug_event_pb2.Execution(op_type=op_type, num_outputs=len(tensors), graph_id=graph_id, input_tensor_ids=input_tensor_ids, output_tensor_ids=output_tensor_ids, output_tensor_device_ids=output_tensor_device_ids, tensor_debug_mode=tensor_debug_mode, code_location=self._process_stack_frames())
            for tensor in tensors:
                if self._should_dump_tensor(op_type, tensor.dtype) and tensor.dtype.is_numpy_compatible:
                    if tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH):
                        if tensor.dtype.is_floating:
                            tensor_proto = _concrete_tensor_to_proto(gen_debug_ops.debug_numeric_summary_v2(tensor, tensor_debug_mode=tensor_debug_mode, output_dtype=dtypes.float64))
                        else:
                            tensor_proto = tensor_pb2.TensorProto()
                    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.SHAPE:
                        if tensor.dtype.is_floating or tensor.dtype.is_integer or tensor.dtype.is_bool:
                            tensor_proto = _concrete_tensor_to_proto(gen_debug_ops.debug_numeric_summary_v2(tensor, tensor_debug_mode=tensor_debug_mode, output_dtype=dtypes.float64))
                        else:
                            tensor_proto = tensor_pb2.TensorProto()
                    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
                        tensor_proto = _concrete_tensor_to_proto(tensor)
                    if tensor_proto:
                        execution_proto.tensor_protos.append(tensor_proto)
            return execution_proto
        else:
            raise NotImplementedError('Tensor instrumentation is not implemented for debug mode %s yet ' % self._tensor_debug_mode)

    def callback(self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
        """Op callback for tracing (dumping) a TF program's execution."""
        del attrs
        writer = self.get_writer()
        if graph:
            is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
            context_id = self._get_context_id(graph)
            output_tensor_ids = self._get_symbolic_tensor_ids(len(outputs))
            if op_type in ('Const', 'Placeholder', 'PlaceholderWithDefault'):
                op_name = outputs[0].name.split(':')[0]
            if is_v1_graph_mode:
                for input_tensor in inputs:
                    if input_tensor in self._placeholder_to_debug_tensor and outputs:
                        outputs[0].op._add_control_input(self._placeholder_to_debug_tensor[input_tensor].op)
            graph_op_creation = debug_event_pb2.GraphOpCreation(op_type=op_type, op_name=op_name, graph_name=graph.name if hasattr(graph, 'name') else None, graph_id=context_id, input_names=[self._lookup_tensor_name(input_tensor) for input_tensor in inputs], num_outputs=len(outputs), output_tensor_ids=output_tensor_ids, code_location=self._process_stack_frames())
            writer.WriteGraphOpCreation(graph_op_creation)
            if outputs and compat.as_bytes(op_type) not in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
                return self._instrument_symbolic_tensors(outputs, op_type, op_name, context_id, output_tensor_ids)
        else:
            op_type_bytes = compat.as_bytes(op_type)
            if op_type_bytes == b'DebugNumericSummaryV2':
                return None
            if op_type_bytes in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
                return None
            context_id = self._func_graph_id_from_func_name(op_type)
            input_ids = [t._id for t in inputs]
            output_tensor_device_ids = [writer.RegisterDeviceAndGetId(output.device) for output in outputs] if outputs else []
            writer.WriteExecution(self._dump_eager_tensors(outputs, op_type, input_ids, output_tensor_device_ids, graph_id=context_id))

    def _lookup_tensor_name(self, tensor):
        """Look up the name of a graph tensor.

    This method maps the name of a debugger-generated Identity or
    DebugIdentityV2 tensor to the name of the original instrumented tensor,
    if `tensor` is such a debugger-created tensor.
    Otherwise, it returns the name of `tensor` as is.

    Args:
      tensor: The graph tensor to look up the name for.

    Returns:
      Name of the orignal instrumented tensor as known to the debugger.
    """
        return self._tensor_aliases.get(tensor.name, tensor.name)

    def _func_graph_id_from_func_name(self, op_type):
        """Attempt to get the ID of a FuncGraph based on an op type name.

    Also caches the ID for faster access later.

    Args:
      op_type: Op type string, which may be the name of a function.

    Returns:
      If the op_type name does not fit the pattern of a function name (e.g.,
      one that starts with "__inference_"), `None` is returned immediately.
      Else, if the FuncGraph is found, ID of the underlying FuncGraph is
      returned as a string.
      Else, `None` is returned.
    """
        op_type = compat.as_bytes(op_type)
        if is_op_type_function(op_type):
            if op_type in self._op_type_to_context_id:
                return self._op_type_to_context_id[op_type]
            with self._context_lock:
                for function in self._function_to_graph_id:
                    if function.name == op_type:
                        graph_id = self._function_to_graph_id[function]
                        self._op_type_to_context_id[op_type] = graph_id
                        return graph_id
            return None
        else:
            return None

    def _get_symbolic_tensor_ids(self, num_tensors):
        tensor_ids = []
        if num_tensors:
            with self._symbolic_tensor_counter_lock:
                for _ in range(num_tensors):
                    self._symbolic_tensor_counter += 1
                    tensor_ids.append(self._symbolic_tensor_counter)
        return tensor_ids

    def _should_dump_tensor(self, op_type, dtype):
        """Determine if the given tensor's value will be dumped.

    The determination is made given the configurations such as `op_regex`,
    `tensor_dtypes`.

    Args:
      op_type: Name of the op's type, as a string (e.g., "MatMul").
      dtype: The dtype of the tensor, as a `dtypes.DType` object.

    Returns:
      A bool indicating whether the tensor's value will be dumped.
    """
        should_dump = True
        if self._op_regex:
            should_dump = should_dump and re.match(self._op_regex, op_type)
        if self._tensor_dtypes:
            if isinstance(self._tensor_dtypes, (list, tuple)):
                should_dump = should_dump and any((dtype == dtype_item for dtype_item in self._tensor_dtypes))
            else:
                should_dump = should_dump and self._tensor_dtypes(dtype)
        return should_dump