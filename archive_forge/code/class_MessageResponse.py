from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class MessageResponse(proto.Message):
    """Response containing a list of messages. Upon delivering a
    MessageResponse to the client, the server:

    -  Updates the stream's delivery cursor to one greater than the
       cursor of the last message in the list.
    -  Subtracts the total number of bytes and messages from the tokens
       available to the server.

    Attributes:
        messages (MutableSequence[google.cloud.pubsublite_v1.types.SequencedMessage]):
            Messages from the topic partition.
    """
    messages: MutableSequence[common.SequencedMessage] = proto.RepeatedField(proto.MESSAGE, number=1, message=common.SequencedMessage)