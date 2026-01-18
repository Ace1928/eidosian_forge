from __future__ import unicode_literals
import datetime
import functools
from google.api_core import datetime_helpers
class TimeToDeadlineTimeout(object):
    """A decorator that decreases timeout set for an RPC based on how much time
    has left till its deadline. The deadline is calculated as
    ``now + initial_timeout`` when this decorator is first called for an rpc.

    In other words this decorator implements deadline semantics in terms of a
    sequence of decreasing timeouts t0 > t1 > t2 ... tn >= 0.

    Args:
        timeout (Optional[float]): the timeout (in seconds) to applied to the
            wrapped function. If `None`, the target function is expected to
            never timeout.
    """

    def __init__(self, timeout=None, clock=datetime_helpers.utcnow):
        self._timeout = timeout
        self._clock = clock

    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """
        first_attempt_timestamp = self._clock().timestamp()

        @functools.wraps(func)
        def func_with_timeout(*args, **kwargs):
            """Wrapped function that adds timeout."""
            remaining_timeout = self._timeout
            if remaining_timeout is not None:
                now_timestamp = self._clock().timestamp()
                if now_timestamp - first_attempt_timestamp < 0.001:
                    now_timestamp = first_attempt_timestamp
                time_since_first_attempt = now_timestamp - first_attempt_timestamp
                kwargs['timeout'] = max(0, self._timeout - time_since_first_attempt)
            return func(*args, **kwargs)
        return func_with_timeout

    def __str__(self):
        return '<TimeToDeadlineTimeout timeout={:.1f}>'.format(self._timeout)