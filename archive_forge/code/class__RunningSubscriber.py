from asyncio import Future, Queue, ensure_future
from typing import Callable, NamedTuple, Dict, List, Set, Optional
from google.cloud.pubsub_v1.subscriber.message import Message
from google.cloud.pubsublite.cloudpubsub.reassignment_handler import ReassignmentHandler
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.internal.wait_ignore_cancelled import (
from google.cloud.pubsublite.internal.wire.assigner import Assigner
from google.cloud.pubsublite.internal.wire.permanent_failable import PermanentFailable
from google.cloud.pubsublite.types import Partition
class _RunningSubscriber(NamedTuple):
    subscriber: AsyncSingleSubscriber
    poller: Future