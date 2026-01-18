import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework
class TensorBoardDebugWrapperSession(GrpcDebugWrapperSession):
    """A tfdbg Session wrapper that can be used with TensorBoard Debugger Plugin.

  This wrapper is the same as `GrpcDebugWrapperSession`, except that it uses a
    predefined `watch_fn` that
    1) uses `DebugIdentity` debug ops with the `gated_grpc` attribute set to
        `True` to allow the interactive enabling and disabling of tensor
       breakpoints.
    2) watches all tensors in the graph.
  This saves the need for the user to define a `watch_fn`.
  """

    def __init__(self, sess, grpc_debug_server_addresses, thread_name_filter=None, send_traceback_and_source_code=True):
        """Constructor of TensorBoardDebugWrapperSession.

    Args:
      sess: The `tf.compat.v1.Session` instance to be wrapped.
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
        super().__init__(sess, grpc_debug_server_addresses, watch_fn=_gated_grpc_watch_fn, thread_name_filter=thread_name_filter)
        self._send_traceback_and_source_code = send_traceback_and_source_code
        self._sent_graph_version = -1
        register_signal_handler()

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None, callable_runner=None, callable_runner_args=None, callable_options=None):
        if self._send_traceback_and_source_code:
            self._sent_graph_version = publish_traceback(self._grpc_debug_server_urls, self.graph, feed_dict, fetches, self._sent_graph_version)
        return super().run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata, callable_runner=callable_runner, callable_runner_args=callable_runner_args, callable_options=callable_options)