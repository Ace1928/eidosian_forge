from concurrent import futures
import logging
from typing import NamedTuple, Callable
from google.cloud.pubsub_v1.subscriber.message import Message
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
class AckId(NamedTuple):
    generation: int
    offset: int

    def encode(self) -> str:
        return str(self.generation) + ',' + str(self.offset)