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
@enum.unique
class LocalConnectionType(enum.Enum):
    """Types of local connection for local credential creation.

    Attributes:
      UDS: Unix domain socket connections
      LOCAL_TCP: Local TCP connections.
    """
    UDS = _cygrpc.LocalConnectionType.uds
    LOCAL_TCP = _cygrpc.LocalConnectionType.local_tcp