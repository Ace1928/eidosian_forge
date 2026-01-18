import functools
import sys
import threading
import time
import typing as t
import warnings
from abc import ABC, abstractmethod
from concurrent import futures
from inspect import iscoroutinefunction
from .retry import retry_base  # noqa
from .retry import retry_all  # noqa
from .retry import retry_always  # noqa
from .retry import retry_any  # noqa
from .retry import retry_if_exception  # noqa
from .retry import retry_if_exception_type  # noqa
from .retry import retry_if_exception_cause_type  # noqa
from .retry import retry_if_not_exception_type  # noqa
from .retry import retry_if_not_result  # noqa
from .retry import retry_if_result  # noqa
from .retry import retry_never  # noqa
from .retry import retry_unless_exception_type  # noqa
from .retry import retry_if_exception_message  # noqa
from .retry import retry_if_not_exception_message  # noqa
from .nap import sleep  # noqa
from .nap import sleep_using_event  # noqa
from .stop import stop_after_attempt  # noqa
from .stop import stop_after_delay  # noqa
from .stop import stop_all  # noqa
from .stop import stop_any  # noqa
from .stop import stop_never  # noqa
from .stop import stop_when_event_set  # noqa
from .wait import wait_chain  # noqa
from .wait import wait_combine  # noqa
from .wait import wait_exponential  # noqa
from .wait import wait_fixed  # noqa
from .wait import wait_incrementing  # noqa
from .wait import wait_none  # noqa
from .wait import wait_random  # noqa
from .wait import wait_random_exponential  # noqa
from .wait import wait_random_exponential as wait_full_jitter  # noqa
from .wait import wait_exponential_jitter  # noqa
from .before import before_log  # noqa
from .before import before_nothing  # noqa
from .after import after_log  # noqa
from .after import after_nothing  # noqa
from .before_sleep import before_sleep_log  # noqa
from .before_sleep import before_sleep_nothing  # noqa
from pip._vendor.tenacity._asyncio import AsyncRetrying  # noqa:E402,I100
class BaseRetrying(ABC):

    def __init__(self, sleep: t.Callable[[t.Union[int, float]], None]=sleep, stop: 'StopBaseT'=stop_never, wait: 'WaitBaseT'=wait_none(), retry: 'RetryBaseT'=retry_if_exception_type(), before: t.Callable[['RetryCallState'], None]=before_nothing, after: t.Callable[['RetryCallState'], None]=after_nothing, before_sleep: t.Optional[t.Callable[['RetryCallState'], None]]=None, reraise: bool=False, retry_error_cls: t.Type[RetryError]=RetryError, retry_error_callback: t.Optional[t.Callable[['RetryCallState'], t.Any]]=None):
        self.sleep = sleep
        self.stop = stop
        self.wait = wait
        self.retry = retry
        self.before = before
        self.after = after
        self.before_sleep = before_sleep
        self.reraise = reraise
        self._local = threading.local()
        self.retry_error_cls = retry_error_cls
        self.retry_error_callback = retry_error_callback

    def copy(self, sleep: t.Union[t.Callable[[t.Union[int, float]], None], object]=_unset, stop: t.Union['StopBaseT', object]=_unset, wait: t.Union['WaitBaseT', object]=_unset, retry: t.Union[retry_base, object]=_unset, before: t.Union[t.Callable[['RetryCallState'], None], object]=_unset, after: t.Union[t.Callable[['RetryCallState'], None], object]=_unset, before_sleep: t.Union[t.Optional[t.Callable[['RetryCallState'], None]], object]=_unset, reraise: t.Union[bool, object]=_unset, retry_error_cls: t.Union[t.Type[RetryError], object]=_unset, retry_error_callback: t.Union[t.Optional[t.Callable[['RetryCallState'], t.Any]], object]=_unset) -> 'BaseRetrying':
        """Copy this object with some parameters changed if needed."""
        return self.__class__(sleep=_first_set(sleep, self.sleep), stop=_first_set(stop, self.stop), wait=_first_set(wait, self.wait), retry=_first_set(retry, self.retry), before=_first_set(before, self.before), after=_first_set(after, self.after), before_sleep=_first_set(before_sleep, self.before_sleep), reraise=_first_set(reraise, self.reraise), retry_error_cls=_first_set(retry_error_cls, self.retry_error_cls), retry_error_callback=_first_set(retry_error_callback, self.retry_error_callback))

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} object at 0x{id(self):x} (stop={self.stop}, wait={self.wait}, sleep={self.sleep}, retry={self.retry}, before={self.before}, after={self.after})>'

    @property
    def statistics(self) -> t.Dict[str, t.Any]:
        """Return a dictionary of runtime statistics.

        This dictionary will be empty when the controller has never been
        ran. When it is running or has ran previously it should have (but
        may not) have useful and/or informational keys and values when
        running is underway and/or completed.

        .. warning:: The keys in this dictionary **should** be some what
                     stable (not changing), but there existence **may**
                     change between major releases as new statistics are
                     gathered or removed so before accessing keys ensure that
                     they actually exist and handle when they do not.

        .. note:: The values in this dictionary are local to the thread
                  running call (so if multiple threads share the same retrying
                  object - either directly or indirectly) they will each have
                  there own view of statistics they have collected (in the
                  future we may provide a way to aggregate the various
                  statistics from each thread).
        """
        try:
            return self._local.statistics
        except AttributeError:
            self._local.statistics = t.cast(t.Dict[str, t.Any], {})
            return self._local.statistics

    def wraps(self, f: WrappedFn) -> WrappedFn:
        """Wrap a function for retrying.

        :param f: A function to wraps for retrying.
        """

        @functools.wraps(f)
        def wrapped_f(*args: t.Any, **kw: t.Any) -> t.Any:
            return self(f, *args, **kw)

        def retry_with(*args: t.Any, **kwargs: t.Any) -> WrappedFn:
            return self.copy(*args, **kwargs).wraps(f)
        wrapped_f.retry = self
        wrapped_f.retry_with = retry_with
        return wrapped_f

    def begin(self) -> None:
        self.statistics.clear()
        self.statistics['start_time'] = time.monotonic()
        self.statistics['attempt_number'] = 1
        self.statistics['idle_for'] = 0

    def iter(self, retry_state: 'RetryCallState') -> t.Union[DoAttempt, DoSleep, t.Any]:
        fut = retry_state.outcome
        if fut is None:
            if self.before is not None:
                self.before(retry_state)
            return DoAttempt()
        is_explicit_retry = fut.failed and isinstance(fut.exception(), TryAgain)
        if not (is_explicit_retry or self.retry(retry_state)):
            return fut.result()
        if self.after is not None:
            self.after(retry_state)
        self.statistics['delay_since_first_attempt'] = retry_state.seconds_since_start
        if self.stop(retry_state):
            if self.retry_error_callback:
                return self.retry_error_callback(retry_state)
            retry_exc = self.retry_error_cls(fut)
            if self.reraise:
                raise retry_exc.reraise()
            raise retry_exc from fut.exception()
        if self.wait:
            sleep = self.wait(retry_state)
        else:
            sleep = 0.0
        retry_state.next_action = RetryAction(sleep)
        retry_state.idle_for += sleep
        self.statistics['idle_for'] += sleep
        self.statistics['attempt_number'] += 1
        if self.before_sleep is not None:
            self.before_sleep(retry_state)
        return DoSleep(sleep)

    def __iter__(self) -> t.Generator[AttemptManager, None, None]:
        self.begin()
        retry_state = RetryCallState(self, fn=None, args=(), kwargs={})
        while True:
            do = self.iter(retry_state=retry_state)
            if isinstance(do, DoAttempt):
                yield AttemptManager(retry_state=retry_state)
            elif isinstance(do, DoSleep):
                retry_state.prepare_for_next_attempt()
                self.sleep(do)
            else:
                break

    @abstractmethod
    def __call__(self, fn: t.Callable[..., WrappedFnReturnT], *args: t.Any, **kwargs: t.Any) -> WrappedFnReturnT:
        pass