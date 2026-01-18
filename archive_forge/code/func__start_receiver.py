import asyncio
from typing import Optional, Set
import logging
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors
from google.cloud.pubsublite.internal.wire.assigner import Assigner
from google.cloud.pubsublite.internal.wire.retrying_connection import (
from google.api_core.exceptions import FailedPrecondition, GoogleAPICallError
from google.cloud.pubsublite.internal.wire.connection_reinitializer import (
from google.cloud.pubsublite.internal.wire.connection import Connection
from google.cloud.pubsublite.types.partition import Partition
from google.cloud.pubsublite_v1.types import (
def _start_receiver(self):
    assert self._receiver is None
    self._receiver = asyncio.ensure_future(self._receive_loop())