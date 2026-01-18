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
def _handle_ack(self, ack_id: AckId):
    flow_control = FlowControlRequest()
    flow_control._pb.allowed_messages = 1
    flow_control._pb.allowed_bytes = self._messages_by_ack_id[ack_id].size_bytes
    self._underlying.allow_flow(flow_control)
    del self._messages_by_ack_id[ack_id]
    if ack_id.generation == self._ack_generation_id:
        try:
            self._ack_set_tracker.ack(ack_id.offset)
        except GoogleAPICallError as e:
            self.fail(e)