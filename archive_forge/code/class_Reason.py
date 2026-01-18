from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import log_entry
class Reason(proto.Enum):
    """An indicator of why entries were omitted.

            Values:
                REASON_UNSPECIFIED (0):
                    Unexpected default.
                RATE_LIMIT (1):
                    Indicates suppression occurred due to relevant entries being
                    received in excess of rate limits. For quotas and limits,
                    see `Logging API quotas and
                    limits <https://cloud.google.com/logging/quotas#api-limits>`__.
                NOT_CONSUMED (2):
                    Indicates suppression occurred due to the
                    client not consuming responses quickly enough.
            """
    REASON_UNSPECIFIED = 0
    RATE_LIMIT = 1
    NOT_CONSUMED = 2