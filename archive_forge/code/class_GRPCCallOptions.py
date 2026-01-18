import abc
import grpc
class GRPCCallOptions(object):
    """A value encapsulating gRPC-specific options passed on RPC invocation.

    This class and its instances have no supported interface - it exists to
    define the type of its instances and its instances exist to be passed to
    other functions.
    """

    def __init__(self, disable_compression, subcall_of, credentials):
        self.disable_compression = disable_compression
        self.subcall_of = subcall_of
        self.credentials = credentials