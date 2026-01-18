from abc import abstractmethod, ABCMeta
from typing import (
from google.cloud.pubsub_v1.subscriber.futures import StreamingPullFuture
from google.cloud.pubsub_v1.subscriber.message import Message
from google.cloud.pubsublite.types import (
class AsyncSubscriberClientInterface(AsyncContextManager, metaclass=ABCMeta):
    """
    An AsyncSubscriberClientInterface reads messages similar to Google Pub/Sub, but must be used in an
    async context.
    Any subscribe failures are unlikely to succeed if retried.

    Must be used in an `async with` block or have __aenter__() awaited before use.
    """

    @abstractmethod
    async def subscribe(self, subscription: Union[SubscriptionPath, str], per_partition_flow_control_settings: FlowControlSettings, fixed_partitions: Optional[Set[Partition]]=None) -> AsyncIterator[Message]:
        """
        Read messages from a subscription.

        Args:
          subscription: The subscription to subscribe to.
          per_partition_flow_control_settings: The flow control settings for each partition subscribed to. Note that these
              settings apply to each partition individually, not in aggregate.
          fixed_partitions: A fixed set of partitions to subscribe to. If not present, will instead use auto-assignment.

        Returns:
          An AsyncIterator with Messages that must have ack() called on each exactly once.

        Raises:
          GoogleApiCallError: On a permanent failure.
        """
        raise NotImplementedError()