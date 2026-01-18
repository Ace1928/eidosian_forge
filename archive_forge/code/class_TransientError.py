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
class TransientError(Exception):
    """Exception type designated for errors that may only be temporary.

    Can have its own message and/or wrap another exception.
    """

    def __init__(self, msg: Optional[str]=None, exc: Optional[BaseException]=None) -> None:
        super().__init__(msg)
        self.message = msg
        self.exception = exc