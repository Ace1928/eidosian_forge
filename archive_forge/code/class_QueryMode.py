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
class QueryMode(proto.Enum):
    """Mode in which the statement must be processed.

        Values:
            NORMAL (0):
                The default mode. Only the statement results
                are returned.
            PLAN (1):
                This mode returns only the query plan,
                without any results or execution statistics
                information.
            PROFILE (2):
                This mode returns both the query plan and the
                execution statistics along with the results.
        """
    NORMAL = 0
    PLAN = 1
    PROFILE = 2