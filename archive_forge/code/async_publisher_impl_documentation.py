from typing import Mapping, Callable, Optional
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.cloudpubsub.message_transforms import (
from google.cloud.pubsublite.cloudpubsub.internal.single_publisher import (
from google.cloud.pubsublite.internal.wire.publisher import Publisher

        Accepts a factory for a Publisher instead of a Publisher because GRPC asyncio uses the current thread's event
        loop.
        