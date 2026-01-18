from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.cloud import CloudRetry
import random
from functools import wraps
import syslog
import time
def _full_jitter_backoff(retries=10, delay=3, max_delay=60, _random=random):
    """ Implements the "Full Jitter" backoff strategy described here
    https://www.awsarchitectureblog.com/2015/03/backoff.html
    Args:
        retries (int): Maximum number of times to retry a request.
        delay (float): Approximate number of seconds to sleep for the first
            retry.
        max_delay (int): The maximum number of seconds to sleep for any retry.
            _random (random.Random or None): Makes this generator testable by
            allowing developers to explicitly pass in the a seeded Random.
    Returns:
        Callable that returns a generator. This generator yields durations in
        seconds to be used as delays for a full jitter backoff strategy.
    Usage:
        >>> backoff = _full_jitter_backoff(retries=5)
        >>> backoff
        <function backoff_backoff at 0x7f0d939facf8>
        >>> list(backoff())
        [3, 6, 5, 23, 38]
        >>> list(backoff())
        [2, 1, 6, 6, 31]
    """

    def backoff_gen():
        for retry in range(0, retries):
            yield _random.randint(0, min(max_delay, delay * 2 ** retry))
    return backoff_gen