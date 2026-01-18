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
class Priority(proto.Enum):
    """The relative priority for requests. Note that priority is not
        applicable for
        [BeginTransaction][google.spanner.v1.Spanner.BeginTransaction].

        The priority acts as a hint to the Cloud Spanner scheduler and does
        not guarantee priority or order of execution. For example:

        -  Some parts of a write operation always execute at
           ``PRIORITY_HIGH``, regardless of the specified priority. This may
           cause you to see an increase in high priority workload even when
           executing a low priority request. This can also potentially cause
           a priority inversion where a lower priority request will be
           fulfilled ahead of a higher priority request.
        -  If a transaction contains multiple operations with different
           priorities, Cloud Spanner does not guarantee to process the
           higher priority operations first. There may be other constraints
           to satisfy, such as order of operations.

        Values:
            PRIORITY_UNSPECIFIED (0):
                ``PRIORITY_UNSPECIFIED`` is equivalent to ``PRIORITY_HIGH``.
            PRIORITY_LOW (1):
                This specifies that the request is low
                priority.
            PRIORITY_MEDIUM (2):
                This specifies that the request is medium
                priority.
            PRIORITY_HIGH (3):
                This specifies that the request is high
                priority.
        """
    PRIORITY_UNSPECIFIED = 0
    PRIORITY_LOW = 1
    PRIORITY_MEDIUM = 2
    PRIORITY_HIGH = 3