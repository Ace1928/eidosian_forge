import asyncio
from typing import Awaitable, TypeVar, Optional, Callable
from google.api_core.exceptions import GoogleAPICallError, Unknown
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors
@property
def _failure_task(self) -> asyncio.Future:
    """Get the failure task, initializing it lazily, since it needs to be initialized in the event loop."""
    if self._maybe_failure_task is None:
        self._maybe_failure_task = asyncio.Future()
    return self._maybe_failure_task