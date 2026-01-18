import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
def _add_jitter(max_percent_jitter):
    """Wraps a existing strategy and adds jitter to it.

    0% to 100% of the spacing value will be added to this value to ensure
    callbacks do not synchronize.
    """
    if max_percent_jitter > 1 or max_percent_jitter < 0:
        raise ValueError("Invalid 'max_percent_jitter', must be greater or equal to 0.0 and less than or equal to 1.0")

    def wrapper(func):
        rnd = random.SystemRandom()

        @functools.wraps(func)
        def decorator(cb, started_at, finished_at, metrics):
            next_run = func(cb, started_at, finished_at, metrics)
            how_often = cb._periodic_spacing
            jitter = how_often * (rnd.random() * max_percent_jitter)
            return next_run + jitter
        decorator.__name__ += '_with_jitter'
        return decorator
    return wrapper