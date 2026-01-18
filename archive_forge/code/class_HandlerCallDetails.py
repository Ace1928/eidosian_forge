import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
class HandlerCallDetails(abc.ABC):
    """Describes an RPC that has just arrived for service.

    Attributes:
      method: The method name of the RPC.
      invocation_metadata: The :term:`metadata` sent by the client.
    """