from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
def _retry_error_helper(exc: Exception, deadline: float | None, next_sleep: float, error_list: list[Exception], predicate_fn: Callable[[Exception], bool], on_error_fn: Callable[[Exception], None] | None, exc_factory_fn: Callable[[list[Exception], RetryFailureReason, float | None], tuple[Exception, Exception | None]], original_timeout: float | None):
    """
    Shared logic for handling an error for all retry implementations

    - Raises an error on timeout or non-retryable error
    - Calls on_error_fn if provided
    - Logs the error

    Args:
       - exc: the exception that was raised
       - deadline: the deadline for the retry, calculated as a diff from time.monotonic()
       - next_sleep: the next sleep interval
       - error_list: the list of exceptions that have been raised so far
       - predicate_fn: takes `exc` and returns true if the operation should be retried
       - on_error_fn: callback to execute when a retryable error occurs
       - exc_factory_fn: callback used to build the exception to be raised on terminal failure
       - original_timeout_val: the original timeout value for the retry (in seconds),
           to be passed to the exception factory for building an error message
    """
    error_list.append(exc)
    if not predicate_fn(exc):
        final_exc, source_exc = exc_factory_fn(error_list, RetryFailureReason.NON_RETRYABLE_ERROR, original_timeout)
        raise final_exc from source_exc
    if on_error_fn is not None:
        on_error_fn(exc)
    if deadline is not None and time.monotonic() + next_sleep > deadline:
        final_exc, source_exc = exc_factory_fn(error_list, RetryFailureReason.TIMEOUT, original_timeout)
        raise final_exc from source_exc
    _LOGGER.debug('Retrying due to {}, sleeping {:.1f}s ...'.format(error_list[-1], next_sleep))