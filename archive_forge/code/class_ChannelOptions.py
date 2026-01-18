import copy
import functools
import sys
import warnings
import grpc
from grpc._cython import cygrpc as _cygrpc
class ChannelOptions(object):
    """Indicates a channel option unique to gRPC Python.

    This enumeration is part of an EXPERIMENTAL API.

    Attributes:
      SingleThreadedUnaryStream: Perform unary-stream RPCs on a single thread.
    """
    SingleThreadedUnaryStream = 'SingleThreadedUnaryStream'