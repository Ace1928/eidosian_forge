from collections import OrderedDict
import os
import re
from typing import (
import warnings
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.speech_v1p1beta1 import gapic_version as package_version
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.cloud.speech_v1p1beta1.services.adaptation import pagers
from google.cloud.speech_v1p1beta1.types import cloud_speech_adaptation, resource
from .transports.base import DEFAULT_CLIENT_INFO, AdaptationTransport
from .transports.grpc import AdaptationGrpcTransport
from .transports.grpc_asyncio import AdaptationGrpcAsyncIOTransport
from .transports.rest import AdaptationRestTransport
class AdaptationClientMeta(type):
    """Metaclass for the Adaptation client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['grpc'] = AdaptationGrpcTransport
    _transport_registry['grpc_asyncio'] = AdaptationGrpcAsyncIOTransport
    _transport_registry['rest'] = AdaptationRestTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[AdaptationTransport]:
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