from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class InitialSubscribeRequest(proto.Message):
    """The first request that must be sent on a newly-opened stream.
    The client must wait for the response before sending subsequent
    requests on the stream.

    Attributes:
        subscription (str):
            The subscription from which to receive
            messages.
        partition (int):
            The partition from which to receive messages. Partitions are
            zero indexed, so ``partition`` must be in the range [0,
            topic.num_partitions).
        initial_location (google.cloud.pubsublite_v1.types.SeekRequest):
            Optional. Initial target location within the
            message backlog. If not set, messages will be
            delivered from the commit cursor for the given
            subscription and partition.
    """
    subscription: str = proto.Field(proto.STRING, number=1)
    partition: int = proto.Field(proto.INT64, number=2)
    initial_location: 'SeekRequest' = proto.Field(proto.MESSAGE, number=4, message='SeekRequest')