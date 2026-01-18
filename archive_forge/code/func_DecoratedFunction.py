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
@functools.wraps(f)
def DecoratedFunction(*args, **kwargs):
    retryer = Retryer(max_retrials=max_retrials, max_wait_ms=max_wait_ms, exponential_sleep_multiplier=exponential_sleep_multiplier, jitter_ms=jitter_ms, status_update_func=status_update_func)
    try:
        return retryer.RetryOnException(f, args=args, kwargs=kwargs, should_retry_if=should_retry_if, sleep_ms=sleep_ms)
    except MaxRetrialsException as mre:
        to_reraise = mre.last_result[1]
        exceptions.reraise(to_reraise[1], tb=to_reraise[2])