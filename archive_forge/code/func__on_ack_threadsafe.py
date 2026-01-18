import asyncio
from typing import Callable, List, Dict, NamedTuple
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsub_v1.subscriber.message import Message
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.internal.wire.permanent_failable import adapt_error
from google.cloud.pubsublite.types import FlowControlSettings
from google.cloud.pubsublite.cloudpubsub.internal.ack_set_tracker import AckSetTracker
from google.cloud.pubsublite.cloudpubsub.internal.wrapped_message import (
from google.cloud.pubsublite.cloudpubsub.message_transformer import MessageTransformer
from google.cloud.pubsublite.cloudpubsub.nack_handler import NackHandler
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.internal.wire.permanent_failable import PermanentFailable
from google.cloud.pubsublite.internal.wire.subscriber import Subscriber
from google.cloud.pubsublite.internal.wire.subscriber_reset_handler import (
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
def _on_ack_threadsafe(self, ack_id: AckId, should_ack: bool) -> None:
    """A function called when a message is acked, may happen from any thread."""
    if should_ack:
        self._loop.call_soon_threadsafe(lambda: self._handle_ack(ack_id))
        return
    try:
        sized_message = self._messages_by_ack_id[ack_id]
        self._nack_handler.on_nack(sized_message.message, lambda: self._on_ack_threadsafe(ack_id, True))
    except Exception as e:
        e2 = adapt_error(e)
        self._loop.call_soon_threadsafe(lambda: self.fail(e2))