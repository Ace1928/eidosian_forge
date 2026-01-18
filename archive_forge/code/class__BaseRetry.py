from __future__ import annotations
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import requests.exceptions
from google.api_core import exceptions
from google.auth import exceptions as auth_exceptions
class _BaseRetry(object):
    """
    Base class for retry configuration objects. This class is intended to capture retry
    and backoff configuration that is common to both synchronous and asynchronous retries,
    for both unary and streaming RPCs. It is not intended to be instantiated directly,
    but rather to be subclassed by the various retry configuration classes.
    """

    def __init__(self, predicate: Callable[[Exception], bool]=if_transient_error, initial: float=_DEFAULT_INITIAL_DELAY, maximum: float=_DEFAULT_MAXIMUM_DELAY, multiplier: float=_DEFAULT_DELAY_MULTIPLIER, timeout: Optional[float]=_DEFAULT_DEADLINE, on_error: Optional[Callable[[Exception], Any]]=None, **kwargs: Any) -> None:
        self._predicate = predicate
        self._initial = initial
        self._multiplier = multiplier
        self._maximum = maximum
        self._timeout = kwargs.get('deadline', timeout)
        self._deadline = self._timeout
        self._on_error = on_error

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError('Not implemented in base class')

    @property
    def deadline(self) -> float | None:
        """
        DEPRECATED: use ``timeout`` instead.  Refer to the ``Retry`` class
        documentation for details.
        """
        return self._timeout

    @property
    def timeout(self) -> float | None:
        return self._timeout

    def with_deadline(self, deadline: float | None) -> Self:
        """Return a copy of this retry with the given timeout.

        DEPRECATED: use :meth:`with_timeout` instead. Refer to the ``Retry`` class
        documentation for details.

        Args:
            deadline (float|None): How long to keep retrying, in seconds. If None,
                no timeout is enforced.

        Returns:
            Retry: A new retry instance with the given timeout.
        """
        return self.with_timeout(deadline)

    def with_timeout(self, timeout: float | None) -> Self:
        """Return a copy of this retry with the given timeout.

        Args:
            timeout (float): How long to keep retrying, in seconds. If None,
                no timeout will be enforced.

        Returns:
            Retry: A new retry instance with the given timeout.
        """
        return type(self)(predicate=self._predicate, initial=self._initial, maximum=self._maximum, multiplier=self._multiplier, timeout=timeout, on_error=self._on_error)

    def with_predicate(self, predicate: Callable[[Exception], bool]) -> Self:
        """Return a copy of this retry with the given predicate.

        Args:
            predicate (Callable[Exception]): A callable that should return
                ``True`` if the given exception is retryable.

        Returns:
            Retry: A new retry instance with the given predicate.
        """
        return type(self)(predicate=predicate, initial=self._initial, maximum=self._maximum, multiplier=self._multiplier, timeout=self._timeout, on_error=self._on_error)

    def with_delay(self, initial: Optional[float]=None, maximum: Optional[float]=None, multiplier: Optional[float]=None) -> Self:
        """Return a copy of this retry with the given delay options.

        Args:
            initial (float): The minimum amount of time to delay (in seconds). This must
                be greater than 0. If None, the current value is used.
            maximum (float): The maximum amount of time to delay (in seconds). If None, the
                current value is used.
            multiplier (float): The multiplier applied to the delay. If None, the current
                value is used.

        Returns:
            Retry: A new retry instance with the given delay options.
        """
        return type(self)(predicate=self._predicate, initial=initial if initial is not None else self._initial, maximum=maximum if maximum is not None else self._maximum, multiplier=multiplier if multiplier is not None else self._multiplier, timeout=self._timeout, on_error=self._on_error)

    def __str__(self) -> str:
        return '<{} predicate={}, initial={:.1f}, maximum={:.1f}, multiplier={:.1f}, timeout={}, on_error={}>'.format(type(self).__name__, self._predicate, self._initial, self._maximum, self._multiplier, self._timeout, self._on_error)