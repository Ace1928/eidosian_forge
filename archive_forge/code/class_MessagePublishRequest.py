from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class MessagePublishRequest(proto.Message):
    """Request to publish messages to the topic.

    Attributes:
        messages (MutableSequence[google.cloud.pubsublite_v1.types.PubSubMessage]):
            The messages to publish.
        first_sequence_number (int):
            The sequence number corresponding to the first message in
            ``messages``. Messages within a batch are ordered and the
            sequence numbers of all subsequent messages in the batch are
            assumed to be incremental.

            Sequence numbers are assigned at the message level and the
            first message published in a publisher client session must
            have a sequence number of 0. All messages must have
            contiguous sequence numbers, which uniquely identify the
            messages accepted by the publisher client. Since messages
            are ordered, the client only needs to specify the sequence
            number of the first message in a published batch. The server
            deduplicates messages with the same sequence number from the
            same publisher ``client_id``.
    """
    messages: MutableSequence[common.PubSubMessage] = proto.RepeatedField(proto.MESSAGE, number=1, message=common.PubSubMessage)
    first_sequence_number: int = proto.Field(proto.INT64, number=2)