from abc import ABC, abstractmethod
from typing import Callable
from google.api_core.exceptions import FailedPrecondition
from google.pubsub_v1 import PubsubMessage
class NackHandler(ABC):
    """
    A NackHandler handles calls to the nack() method which is not expressible in Pub/Sub Lite.
    """

    @abstractmethod
    def on_nack(self, message: PubsubMessage, ack: Callable[[], None]):
        """Handle a negative acknowledgement. ack must eventually be called.

        This method will be called on an event loop and should not block.

        Args:
          message: The nacked message.
          ack: A callable to acknowledge the underlying message. This must eventually be called.

        Raises:
          GoogleAPICallError: To fail the client if raised inline.
        """
        pass