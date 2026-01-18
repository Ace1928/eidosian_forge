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
class AuthMetadataContext(abc.ABC):
    """Provides information to call credentials metadata plugins.

    Attributes:
      service_url: A string URL of the service being called into.
      method_name: A string of the fully qualified method name being called.
    """