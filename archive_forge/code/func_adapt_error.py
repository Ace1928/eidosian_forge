import asyncio
from typing import Awaitable, TypeVar, Optional, Callable
from google.api_core.exceptions import GoogleAPICallError, Unknown
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors
def adapt_error(e: Exception) -> GoogleAPICallError:
    if isinstance(e, GoogleAPICallError):
        return e
    return Unknown('Had an unknown error', errors=[e])