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
class Backoff(abc.ABC):
    """A backoff strategy: decides whether to sleep or give up when an exception is raised."""

    @abc.abstractmethod
    def next_sleep_or_reraise(self, exc: Exception) -> datetime.timedelta:
        raise NotImplementedError