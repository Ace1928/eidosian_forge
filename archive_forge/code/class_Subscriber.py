from abc import abstractmethod, ABCMeta
from typing import AsyncContextManager, List
from google.cloud.pubsublite_v1.types import SequencedMessage, FlowControlRequest
class Subscriber(AsyncContextManager, metaclass=ABCMeta):
    """
    A Pub/Sub Lite asynchronous wire protocol subscriber.
    """

    @abstractmethod
    async def read(self) -> List[SequencedMessage.meta.pb]:
        """
        Read a batch of messages off of the stream.

        Returns:
          The next batch of messages.

        Raises:
          GoogleAPICallError: On a permanent error.
        """
        raise NotImplementedError()

    @abstractmethod
    def allow_flow(self, request: FlowControlRequest):
        """
        Allow an additional amount of messages and bytes to be sent to this client.
        """
        raise NotImplementedError()