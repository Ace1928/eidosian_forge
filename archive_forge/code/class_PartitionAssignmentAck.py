from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class PartitionAssignmentAck(proto.Message):
    """Acknowledge receipt and handling of the previous assignment.
    If not sent within a short period after receiving the
    assignment, partitions may remain unassigned for a period of
    time until the client is known to be inactive, after which time
    the server will break the stream.

    """