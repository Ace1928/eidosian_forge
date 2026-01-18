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
from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.cloud.pubsublite_v1.services.admin_service import pagers
from google.cloud.pubsublite_v1.types import admin
from google.cloud.pubsublite_v1.types import common
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from .transports.base import AdminServiceTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import AdminServiceGrpcTransport
from .transports.grpc_asyncio import AdminServiceGrpcAsyncIOTransport
class AdminServiceClientMeta(type):
    """Metaclass for the AdminService client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['grpc'] = AdminServiceGrpcTransport
    _transport_registry['grpc_asyncio'] = AdminServiceGrpcAsyncIOTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[AdminServiceTransport]:
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