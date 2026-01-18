from collections import OrderedDict
import os
import re
from typing import Dict, Mapping, MutableMapping, MutableSequence, Optional, Iterable, Sequence, Tuple, Type, Union, cast
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials             # type: ignore
from google.auth.transport import mtls                            # type: ignore
from google.auth.transport.grpc import SslCredentials             # type: ignore
from google.auth.exceptions import MutualTLSChannelError          # type: ignore
from google.oauth2 import service_account                         # type: ignore
from cloudsdk.google.protobuf import struct_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.services.spanner import pagers
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import commit_response
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import mutation
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import result_set
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import spanner
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import transaction
from .transports.base import SpannerTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import SpannerGrpcTransport
from .transports.grpc_asyncio import SpannerGrpcAsyncIOTransport
from .transports.rest import SpannerRestTransport
class SpannerClientMeta(type):
    """Metaclass for the Spanner client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """
    _transport_registry = OrderedDict()
    _transport_registry['grpc'] = SpannerGrpcTransport
    _transport_registry['grpc_asyncio'] = SpannerGrpcAsyncIOTransport
    _transport_registry['rest'] = SpannerRestTransport

    def get_transport_class(cls, label: Optional[str]=None) -> Type[SpannerTransport]:
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