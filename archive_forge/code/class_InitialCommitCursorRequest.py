from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class InitialCommitCursorRequest(proto.Message):
    """The first streaming request that must be sent on a
    newly-opened stream. The client must wait for the response
    before sending subsequent requests on the stream.

    Attributes:
        subscription (str):
            The subscription for which to manage
            committed cursors.
        partition (int):
            The partition for which to manage committed cursors.
            Partitions are zero indexed, so ``partition`` must be in the
            range [0, topic.num_partitions).
    """
    subscription: str = proto.Field(proto.STRING, number=1)
    partition: int = proto.Field(proto.INT64, number=2)