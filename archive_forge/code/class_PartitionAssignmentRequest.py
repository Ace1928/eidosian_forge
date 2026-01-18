from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class PartitionAssignmentRequest(proto.Message):
    """A request on the PartitionAssignment stream.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        initial (google.cloud.pubsublite_v1.types.InitialPartitionAssignmentRequest):
            Initial request on the stream.

            This field is a member of `oneof`_ ``request``.
        ack (google.cloud.pubsublite_v1.types.PartitionAssignmentAck):
            Acknowledgement of a partition assignment.

            This field is a member of `oneof`_ ``request``.
    """
    initial: 'InitialPartitionAssignmentRequest' = proto.Field(proto.MESSAGE, number=1, oneof='request', message='InitialPartitionAssignmentRequest')
    ack: 'PartitionAssignmentAck' = proto.Field(proto.MESSAGE, number=2, oneof='request', message='PartitionAssignmentAck')