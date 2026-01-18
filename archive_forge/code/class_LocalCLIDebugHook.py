from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook
class LocalCLIDebugHook(session_run_hook.SessionRunHook):
    """Command-line-interface debugger hook.

  Can be used as a hook for `tf.compat.v1.train.MonitoredSession`s and
  `tf.estimator.Estimator`s. Provides a substitute for
  `tfdbg.LocalCLIDebugWrapperSession` in cases where the session is not directly
  available.
  """

    def __init__(self, ui_type='readline', dump_root=None, thread_name_filter=None, config_file_path=None):
        """Create a local debugger command-line interface (CLI) hook.

    Args:
      ui_type: (`str`) requested user-interface type. Currently supported:
        (readline).
      dump_root: (`str`) optional path to the dump root directory. Must be a
        directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        `run()` calls and removed afterwards.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      config_file_path: Optional override to the default configuration file
        path, which is at `${HOME}/.tfdbg_config`.
    """
        self._ui_type = ui_type
        self._dump_root = dump_root
        self._thread_name_filter = thread_name_filter
        self._session_wrapper = None
        self._pending_tensor_filters = {}
        self._config_file_path = config_file_path

    def add_tensor_filter(self, filter_name, tensor_filter):
        """Add a tensor filter.

    See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    Override default behavior to accommodate the possibility of this method
    being
    called prior to the initialization of the underlying
    `LocalCLIDebugWrapperSession` object.

    Args:
      filter_name: See doc of `LocalCLIDebugWrapperSession.add_tensor_filter()`
        for details.
      tensor_filter: See doc of
        `LocalCLIDebugWrapperSession.add_tensor_filter()` for details.
    """
        if self._session_wrapper:
            self._session_wrapper.add_tensor_filter(filter_name, tensor_filter)
        else:
            self._pending_tensor_filters[filter_name] = tensor_filter

    def begin(self):
        pass

    def before_run(self, run_context):
        if not self._session_wrapper:
            self._session_wrapper = local_cli_wrapper.LocalCLIDebugWrapperSession(run_context.session, ui_type=self._ui_type, dump_root=self._dump_root, thread_name_filter=self._thread_name_filter, config_file_path=self._config_file_path)
            for filter_name in self._pending_tensor_filters:
                self._session_wrapper.add_tensor_filter(filter_name, self._pending_tensor_filters[filter_name])
        self._session_wrapper.increment_run_call_count()
        on_run_start_request = framework.OnRunStartRequest(run_context.original_args.fetches, run_context.original_args.feed_dict, None, None, self._session_wrapper.run_call_count)
        on_run_start_response = self._session_wrapper.on_run_start(on_run_start_request)
        self._performed_action = on_run_start_response.action
        run_args = session_run_hook.SessionRunArgs(None, feed_dict=None, options=config_pb2.RunOptions())
        if self._performed_action == framework.OnRunStartAction.DEBUG_RUN:
            self._session_wrapper._decorate_run_options_for_debug(run_args.options, on_run_start_response.debug_urls, debug_ops=on_run_start_response.debug_ops, node_name_regex_allowlist=on_run_start_response.node_name_regex_allowlist, op_type_regex_allowlist=on_run_start_response.op_type_regex_allowlist, tensor_dtype_regex_allowlist=on_run_start_response.tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=on_run_start_response.tolerate_debug_op_creation_failures)
        elif self._performed_action == framework.OnRunStartAction.PROFILE_RUN:
            self._session_wrapper._decorate_run_options_for_profile(run_args.options)
        return run_args

    def after_run(self, run_context, run_values):
        on_run_end_request = framework.OnRunEndRequest(self._performed_action, run_values.run_metadata)
        self._session_wrapper.on_run_end(on_run_end_request)