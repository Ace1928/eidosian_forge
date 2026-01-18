from typing import Optional, List, Union
from google.api_core.client_options import ClientOptions
from google.api_core.operation import Operation
from google.auth.credentials import Credentials
from cloudsdk.google.protobuf.field_mask_pb2 import FieldMask  # pytype: disable=pyi-error
from google.cloud.pubsublite.admin_client_interface import AdminClientInterface
from google.cloud.pubsublite.internal.constructable_from_service_account import (
from google.cloud.pubsublite.internal.endpoints import regional_endpoint
from google.cloud.pubsublite.internal.wire.admin_client_impl import AdminClientImpl
from google.cloud.pubsublite.types import (
from google.cloud.pubsublite.types.paths import ReservationPath
from google.cloud.pubsublite_v1 import (

        Create a new AdminClient.

        Args:
            region: The cloud region to connect to.
            credentials: The credentials to use when connecting.
            transport: The transport to use.
            client_options: The client options to use when connecting. If used, must explicitly set `api_endpoint`.
        