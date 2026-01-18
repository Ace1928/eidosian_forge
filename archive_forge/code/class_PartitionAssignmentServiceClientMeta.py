from collections import OrderedDict
import os
import re
from typing import (
from google.cloud.pubsublite_v1 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.pubsublite_v1.types import subscriber
from google.longrunning import operations_pb2
from .transports.base import PartitionAssignmentServiceTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import PartitionAssignmentServiceGrpcTransport
from .transports.grpc_asyncio import PartitionAssignmentServiceGrpcAsyncIOTransport
class PartitionAssignmentServiceClientMeta(type):
    """Metaclass for the PartitionAssignmentService client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['grpc'] = PartitionAssignmentServiceGrpcTransport
    _transport_registry['grpc_asyncio'] = PartitionAssignmentServiceGrpcAsyncIOTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[PartitionAssignmentServiceTransport]:
        """Returns an appropriate transport class.

        Args:
            label: The name of the desired transport. If none is
                provided, then the first transport in the registry is used.

        Returns:
            The transport class to use.
        """
        if label:
            return cls._transport_registry[label]
        return next(iter(cls._transport_registry.values()))