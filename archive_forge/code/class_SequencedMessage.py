from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class SequencedMessage(proto.Message):
    """A message that has been stored and sequenced by the Pub/Sub
    Lite system.

    Attributes:
        cursor (google.cloud.pubsublite_v1.types.Cursor):
            The position of a message within the
            partition where it is stored.
        publish_time (google.protobuf.timestamp_pb2.Timestamp):
            The time when the message was received by the
            server when it was first published.
        message (google.cloud.pubsublite_v1.types.PubSubMessage):
            The user message.
        size_bytes (int):
            The size in bytes of this message for flow
            control and quota purposes.
    """
    cursor: 'Cursor' = proto.Field(proto.MESSAGE, number=1, message='Cursor')
    publish_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    message: 'PubSubMessage' = proto.Field(proto.MESSAGE, number=3, message='PubSubMessage')
    size_bytes: int = proto.Field(proto.INT64, number=4)