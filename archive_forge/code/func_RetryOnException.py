from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import functools
import itertools
import math
import random
import sys
import time
from googlecloudsdk.core import exceptions
def RetryOnException(self, func, args=None, kwargs=None, should_retry_if=None, sleep_ms=None):
    """Retries the function if an exception occurs.

    Args:
      func: The function to call and retry.
      args: a sequence of positional arguments to be passed to func.
      kwargs: a dictionary of positional arguments to be passed to func.
      should_retry_if: func(exc_type, exc_value, exc_traceback, state) that
          returns True or False.
      sleep_ms: int or iterable for how long to wait between trials.

    Returns:
      Whatever the function returns.

    Raises:
      RetryException, WaitException: if function is retries too many times,
        or time limit is reached.
    """
    args = args if args is not None else ()
    kwargs = kwargs if kwargs is not None else {}

    def TryFunc():
        try:
            return (func(*args, **kwargs), None)
        except:
            return (None, sys.exc_info())
    if should_retry_if is None:
        should_retry = lambda x, s: x[1] is not None
    else:

        def ShouldRetryFunc(try_func_result, state):
            exc_info = try_func_result[1]
            if exc_info is None:
                return False
            return should_retry_if(exc_info[0], exc_info[1], exc_info[2], state)
        should_retry = ShouldRetryFunc
    result, exc_info = self.RetryOnResult(TryFunc, should_retry_if=should_retry, sleep_ms=sleep_ms)
    if exc_info:
        exceptions.reraise(exc_info[1], tb=exc_info[2])
    return result