from typing import Sequence
import grpc
from grpc.aio._server import Server
def add_generic_rpc_handlers(self, generic_rpc_handlers: Sequence[grpc.GenericRpcHandler]):
    """Override generic_rpc_handlers before adding to the gRPC server.

        This function will override all user defined handlers to have
            1. None `response_serializer` so the server can pass back the
            raw protobuf bytes to the user.
            2. `unary_unary` is always calling the unary function generated via
            `self.service_handler_factory`
            3. `unary_stream` is always calling the streaming function generated via
            `self.service_handler_factory`
        """
    serve_rpc_handlers = {}
    rpc_handler = generic_rpc_handlers[0]
    for service_method, method_handler in rpc_handler._method_handlers.items():
        serve_method_handler = method_handler._replace(response_serializer=None, unary_unary=self.service_handler_factory(service_method=service_method, stream=False), unary_stream=self.service_handler_factory(service_method=service_method, stream=True))
        serve_rpc_handlers[service_method] = serve_method_handler
    generic_rpc_handlers[0]._method_handlers = serve_rpc_handlers
    self.generic_rpc_handlers.append(generic_rpc_handlers)
    super().add_generic_rpc_handlers(generic_rpc_handlers)