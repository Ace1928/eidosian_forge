from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import struct_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import keys
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import mutation
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import result_set
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import transaction as gs_transaction
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import type as gs_type
class DeleteSessionRequest(proto.Message):
    """The request for
    [DeleteSession][google.spanner.v1.Spanner.DeleteSession].

    Attributes:
        name (str):
            Required. The name of the session to delete.
    """
    name: str = proto.Field(proto.STRING, number=1)