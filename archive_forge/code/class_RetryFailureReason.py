from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
class RetryFailureReason(Enum):
    """
    The cause of a failed retry, used when building exceptions
    """
    TIMEOUT = 0
    NON_RETRYABLE_ERROR = 1