from concurrent import futures
import logging
from typing import NamedTuple, Callable
from google.cloud.pubsub_v1.subscriber.message import Message
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
def ack_with_response(self) -> 'futures.Future':
    self._ack_handler(self._id, True)
    return _SUCCESS_FUTURE