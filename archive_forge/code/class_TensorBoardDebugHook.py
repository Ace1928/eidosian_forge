from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import grpc_wrapper
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.training import session_run_hook
class TensorBoardDebugHook(GrpcDebugHook):
    """A tfdbg hook that can be used with TensorBoard Debugger Plugin.

  This hook is the same as `GrpcDebugHook`, except that it uses a predefined
    `watch_fn` that
    1) uses `DebugIdentity` debug ops with the `gated_grpc` attribute set to
        `True`, to allow the interactive enabling and disabling of tensor
       breakpoints.
    2) watches all tensors in the graph.
  This saves the need for the user to define a `watch_fn`.
  """

    def __init__(self, grpc_debug_server_addresses, thread_name_filter=None, send_traceback_and_source_code=True):
        """Constructor of TensorBoardDebugHook.

    Args:
      grpc_debug_server_addresses: gRPC address(es) of debug server(s), as a
        `str` or a `list` of `str`s. E.g., "localhost:2333",
        "grpc://localhost:2333", ["192.168.0.7:2333", "192.168.0.8:2333"].
      thread_name_filter: Optional filter for thread names.
      send_traceback_and_source_code: Whether traceback of graph elements and
        the source code are to be sent to the debug server(s).
    """

        def _gated_grpc_watch_fn(fetches, feeds):
            del fetches, feeds
            return framework.WatchOptions(debug_ops=['DebugIdentity(gated_grpc=true)'])
        super(TensorBoardDebugHook, self).__init__(grpc_debug_server_addresses, watch_fn=_gated_grpc_watch_fn, thread_name_filter=thread_name_filter)
        self._grpc_debug_server_addresses = grpc_debug_server_addresses
        self._send_traceback_and_source_code = send_traceback_and_source_code
        self._sent_graph_version = -1
        grpc_wrapper.register_signal_handler()

    def before_run(self, run_context):
        if self._send_traceback_and_source_code:
            self._sent_graph_version = grpc_wrapper.publish_traceback(self._grpc_debug_server_addresses, run_context.session.graph, run_context.original_args.feed_dict, run_context.original_args.fetches, self._sent_graph_version)
        return super(TensorBoardDebugHook, self).before_run(run_context)