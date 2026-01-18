from abc import ABC, abstractmethod
from typing import Callable
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite_v1 import SequencedMessage
class MessageTransformer(ABC):
    """
    A MessageTransformer turns Pub/Sub Lite message protos into Pub/Sub message protos.
    """

    @abstractmethod
    def transform(self, source: SequencedMessage) -> PubsubMessage:
        """Transform a SequencedMessage to a PubsubMessage.

        Args:
          source: The message to transform.

        Raises:
          GoogleAPICallError: To fail the client if raised inline.
        """
        pass

    @staticmethod
    def of_callable(transformer: Callable[[SequencedMessage], PubsubMessage]):

        class CallableTransformer(MessageTransformer):

            def transform(self, source: SequencedMessage) -> PubsubMessage:
                return transformer(source)
        return CallableTransformer()