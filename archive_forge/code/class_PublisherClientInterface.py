from abc import abstractmethod, ABCMeta
from concurrent.futures import Future
from typing import ContextManager, Mapping, Union, AsyncContextManager
from google.cloud.pubsublite.types import TopicPath
class PublisherClientInterface(ContextManager, metaclass=ABCMeta):
    """
    A PublisherClientInterface publishes messages similar to Google Pub/Sub.
    Any publish failures are unlikely to succeed if retried.

    Must be used in a `with` block or have __enter__() called before use.
    """

    @abstractmethod
    def publish(self, topic: Union[TopicPath, str], data: bytes, ordering_key: str='', **attrs: Mapping[str, str]) -> 'Future[str]':
        """
        Publish a message.

        Args:
          topic: The topic to publish to. Publishes to new topics may have nontrivial startup latency.
          data: The bytestring payload of the message
          ordering_key: The key to enforce ordering on, or "" for no ordering.
          **attrs: Additional attributes to send.

        Returns:
          A future completed with an ack id of type str, which can be decoded using
          MessageMetadata.decode.

        Raises:
          GoogleApiCallError: On a permanent failure.
        """
        raise NotImplementedError()