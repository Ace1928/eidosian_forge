from concurrent.futures import Future
from typing import Optional, Mapping, Union
from uuid import uuid4
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsub_v1.types import BatchSettings
from google.cloud.pubsublite.cloudpubsub.internal.make_publisher import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_async_publisher_client import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_publisher_client import (
from google.cloud.pubsublite.cloudpubsub.publisher_client_interface import (
from google.cloud.pubsublite.internal.constructable_from_service_account import (
from google.cloud.pubsublite.internal.publisher_client_id import PublisherClientId
from google.cloud.pubsublite.internal.require_started import RequireStarted
from google.cloud.pubsublite.internal.wire.make_publisher import (
from google.cloud.pubsublite.types import TopicPath
def _get_client_id(enable_idempotence: bool):
    return PublisherClientId(uuid4().bytes) if enable_idempotence else None