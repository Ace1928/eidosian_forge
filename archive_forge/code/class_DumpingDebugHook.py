from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook
class DumpingDebugHook(session_run_hook.SessionRunHook):
    """A debugger hook that dumps debug data to filesystem.

  Can be used as a hook for `tf.compat.v1.train.MonitoredSession`s and
  `tf.estimator.Estimator`s.
  """

    def __init__(self, session_root, watch_fn=None, thread_name_filter=None):
        """Create a local debugger command-line interface (CLI) hook.

    Args:
      session_root: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      watch_fn: See doc of
        `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
    """
        self._session_root = session_root
        self._watch_fn = watch_fn
        self._thread_name_filter = thread_name_filter
        self._session_wrapper = None

    def begin(self):
        pass

    def before_run(self, run_context):
        reset_disk_byte_usage = False
        if not self._session_wrapper:
            self._session_wrapper = dumping_wrapper.DumpingDebugWrapperSession(run_context.session, self._session_root, watch_fn=self._watch_fn, thread_name_filter=self._thread_name_filter)
            reset_disk_byte_usage = True
        self._session_wrapper.increment_run_call_count()
        debug_urls, watch_options = self._session_wrapper._prepare_run_watch_config(run_context.original_args.fetches, run_context.original_args.feed_dict)
        run_options = config_pb2.RunOptions()
        debug_utils.watch_graph(run_options, run_context.session.graph, debug_urls=debug_urls, debug_ops=watch_options.debug_ops, node_name_regex_allowlist=watch_options.node_name_regex_allowlist, op_type_regex_allowlist=watch_options.op_type_regex_allowlist, tensor_dtype_regex_allowlist=watch_options.tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=watch_options.tolerate_debug_op_creation_failures, reset_disk_byte_usage=reset_disk_byte_usage)
        run_args = session_run_hook.SessionRunArgs(None, feed_dict=None, options=run_options)
        return run_args

    def after_run(self, run_context, run_values):
        pass