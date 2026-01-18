import concurrent.futures
import threading
from concurrent.futures.thread import ThreadPoolExecutor
from typing import ContextManager, Optional
from google.api_core.exceptions import GoogleAPICallError
from functools import partial
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors
from google.cloud.pubsublite.cloudpubsub.internal.managed_event_loop import (
from google.cloud.pubsublite.cloudpubsub.internal.streaming_pull_manager import (
from google.cloud.pubsublite.cloudpubsub.internal.single_subscriber import (
from google.cloud.pubsublite.cloudpubsub.subscriber_client_interface import (
def add_close_callback(self, close_callback: CloseCallback):
    """
        A close callback must be set exactly once by the StreamingPullFuture managing this subscriber.

        This two-phase init model is made necessary by the requirements of StreamingPullFuture.
        """
    with self._close_lock:
        assert self._close_callback is None
        self._close_callback = close_callback