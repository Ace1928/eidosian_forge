from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class StreamingCommitCursorRequest(proto.Message):
    """A request sent from the client to the server on a stream.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        initial (google.cloud.pubsublite_v1.types.InitialCommitCursorRequest):
            Initial request on the stream.

            This field is a member of `oneof`_ ``request``.
        commit (google.cloud.pubsublite_v1.types.SequencedCommitCursorRequest):
            Request to commit a new cursor value.

            This field is a member of `oneof`_ ``request``.
    """
    initial: 'InitialCommitCursorRequest' = proto.Field(proto.MESSAGE, number=1, oneof='request', message='InitialCommitCursorRequest')
    commit: 'SequencedCommitCursorRequest' = proto.Field(proto.MESSAGE, number=2, oneof='request', message='SequencedCommitCursorRequest')