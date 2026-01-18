import abc
import asyncio
import datetime
import functools
import logging
import os
import random
import threading
import time
from typing import Any, Awaitable, Callable, Generic, Optional, Tuple, Type, TypeVar
from requests import HTTPError
import wandb
from wandb.util import CheckRetryFnType
from .mailbox import ContextCancelledError
def _sleep_check_cancelled(self, wait_seconds: float, cancel_event: Optional[threading.Event]) -> bool:
    if not cancel_event:
        SLEEP_FN(wait_seconds)
        return False
    cancelled = cancel_event.wait(wait_seconds)
    return cancelled