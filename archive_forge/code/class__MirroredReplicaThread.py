import contextlib
import threading
import weakref
from tensorflow.python import pywrap_tfe
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.util import traceback_utils
class _MirroredReplicaThread(threading.Thread):
    """A thread that runs() a function on a device."""

    def __init__(self, dist, coord, replica_id, devices, variable_creator_fn, fn, caching_scope, args, kwargs, thread_local_callables=None):
        super(_MirroredReplicaThread, self).__init__()
        self.coord = coord
        self.distribution = dist
        self.devices = devices
        self.replica_id = replica_id
        self.replica_id_in_sync_group = dist.extended._get_replica_id_in_sync_group(replica_id)
        self.variable_creator_fn = variable_creator_fn
        self.main_fn = fn
        self.main_args = args
        self.main_kwargs = kwargs
        self.main_result = None
        self.done = False
        self.merge_fn = None
        self.merge_args = None
        self.merge_kwargs = None
        self.merge_result = None
        self.captured_name_scope = None
        self.captured_var_scope = None
        try:
            self.caching_scope_entered = caching_scope.new_cache_scope_count
            self.caching_scope_exited = caching_scope.cache_scope_exited_count
        except AttributeError:
            self.caching_scope_entered = None
            self.caching_scope_exited = None
        self.should_run = threading.Event()
        self.has_paused = threading.Event()
        context.ensure_initialized()
        ctx = context.context()
        self.in_eager = ctx.executing_eagerly()
        self.record_thread_local_summary_state()
        self.record_thread_local_eager_context_state()
        self.context_device_policy = pywrap_tfe.TFE_ContextGetDevicePlacementPolicy(ctx._context_handle)
        self.graph = ops.get_default_graph()
        with ops.init_scope():
            self._init_in_eager = context.executing_eagerly()
            self._init_graph = ops.get_default_graph()
        self._variable_creator_stack = self.graph._variable_creator_stack[:]
        self._var_scope = variable_scope.get_variable_scope()
        self._name_scope = self.graph.get_name_scope()
        if self._name_scope:
            self._name_scope += '/'
        if self.replica_id > 0:
            if not self._name_scope:
                self._name_scope = ''
            self._name_scope += 'replica_%d/' % self.replica_id
        self._thread_local_callables = thread_local_callables

    def run(self):
        self.should_run.wait()
        self.should_run.clear()
        try:
            if self.coord.should_stop():
                return
            self.restore_thread_local_summary_state()
            self.restore_thread_local_callable()
            self.restore_thread_local_eager_context_state()
            if self.caching_scope_entered is not None and self.caching_scope_exited is not None:
                distribute_utils.caching_scope_local.new_cache_scope_count = self.caching_scope_entered
                distribute_utils.caching_scope_local.cache_scope_exited_count = self.caching_scope_exited
            with self.coord.stop_on_exception(), _enter_graph(self._init_graph, self._init_in_eager), _enter_graph(self.graph, self.in_eager, self._variable_creator_stack), context.device_policy(self.context_device_policy), _MirroredReplicaContext(self.distribution, self.replica_id_in_sync_group), ops.device(self.devices[self.replica_id]), ops.name_scope(self._name_scope), variable_scope.variable_scope(self._var_scope, reuse=self.replica_id > 0), variable_scope.variable_creator_scope(self.variable_creator_fn):
                self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
                self.done = True
        finally:
            self.has_paused.set()

    def record_thread_local_summary_state(self):
        """Record the thread local summary state in self."""
        summary_state = summary_ops_v2._summary_state
        self._summary_step = summary_state.step
        self._summary_writer = summary_state.writer
        self._summary_recording = summary_state.is_recording
        self._summary_recording_distribution_strategy = summary_state.is_recording_distribution_strategy

    def restore_thread_local_summary_state(self):
        """Restore thread local summary state from self."""
        summary_state = summary_ops_v2._summary_state
        summary_state.step = self._summary_step
        summary_state.writer = self._summary_writer
        summary_state.is_recording = self._summary_recording
        summary_state.is_recording_distribution_strategy = self._summary_recording_distribution_strategy

    def record_thread_local_eager_context_state(self):
        ctx = context.context()
        eager_context_state = ctx._thread_local_data
        self._eager_context_op_callbacks = eager_context_state.op_callbacks

    def restore_thread_local_eager_context_state(self):
        ctx = context.context()
        eager_context_state = ctx._thread_local_data
        eager_context_state.op_callbacks = self._eager_context_op_callbacks

    def restore_thread_local_callable(self):
        if self._thread_local_callables:
            for fn in self._thread_local_callables:
                fn()