from typing import NamedTuple
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite_v1.types.common import Cursor
from google.cloud.pubsublite.types.partition import Partition
class MessageMetadata(NamedTuple):
    """Information about a message in Pub/Sub Lite.

    Attributes:
        partition (Partition):
            The partition of the topic that the message was published to.
        cursor (Cursor):
            A cursor containing the offset that the message was assigned.
            If this MessageMetadata was returned for a publish result and
            publish idempotence was enabled, the offset may be -1 when the
            message was identified as a duplicate of an already successfully
            published message, but the server did not have sufficient
            information to return the message's offset at publish time. Messages
            received by subscribers will always have the correct offset.
    """
    partition: Partition
    cursor: Cursor

    def encode(self) -> str:
        return self._encode_parts(self.partition.value, self.cursor._pb.offset)

    @staticmethod
    def _encode_parts(partition: int, offset: int) -> str:
        return fast_serialize.dump([partition, offset])

    @staticmethod
    def decode(source: str) -> 'MessageMetadata':
        loaded = fast_serialize.load(source)
        cursor = Cursor()
        cursor._pb.offset = loaded[1]
        return MessageMetadata(partition=Partition(loaded[0]), cursor=cursor)