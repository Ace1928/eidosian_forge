from concurrent.futures import Future
from typing import Callable, Union, Mapping
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsublite.cloudpubsub.internal.client_multiplexer import (
from google.cloud.pubsublite.cloudpubsub.internal.single_publisher import (
from google.cloud.pubsublite.cloudpubsub.publisher_client_interface import (
from google.cloud.pubsublite.types import TopicPath
def _on_future_completion(self, topic: TopicPath, publisher: SinglePublisher, future: 'Future[str]'):
    try:
        future.result()
    except GoogleAPICallError:
        self._multiplexer.try_erase(topic, publisher)