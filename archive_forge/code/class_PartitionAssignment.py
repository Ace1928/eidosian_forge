from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class PartitionAssignment(proto.Message):
    """PartitionAssignments should not race with acknowledgements.
    There should be exactly one unacknowledged PartitionAssignment
    at a time. If not, the client must break the stream.

    Attributes:
        partitions (MutableSequence[int]):
            The list of partition numbers this subscriber
            is assigned to.
    """
    partitions: MutableSequence[int] = proto.RepeatedField(proto.INT64, number=1)