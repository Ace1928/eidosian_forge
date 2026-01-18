from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, Optional, Set
from threading import Lock
from google.cloud.pubsub_v1.subscriber.futures import StreamingPullFuture
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.cloudpubsub.internal.subscriber_impl import SubscriberImpl
from google.cloud.pubsublite.cloudpubsub.subscriber_client_interface import (
from google.cloud.pubsublite.types import (
@staticmethod
def _cancel_streaming_pull_future(fut: StreamingPullFuture):
    try:
        fut.cancel()
        fut.result()
    except:
        pass