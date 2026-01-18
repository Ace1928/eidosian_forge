from typing import Sequence
import grpc
from grpc.aio._server import Server
class DummyServicer:
    """Dummy servicer for gRPC server to call on.

    This is a dummy class that just pass through when calling on any method.
    User defined servicer function will attempt to add the method on this class to the
    gRPC server, but our gRPC server will override the caller to call gRPCProxy.
    """

    def __getattr__(self, attr):
        pass