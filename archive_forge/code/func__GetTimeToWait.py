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
def _GetTimeToWait(self, last_retrial, sleep_ms):
    """Get time to wait after applying modifyers.

    Apply the exponential sleep multiplyer, jitter and ceiling limiting to the
    base sleep time.

    Args:
      last_retrial: int, which retry attempt we just tried. First try this is 0.
      sleep_ms: int, how long to wait between the current trials.

    Returns:
      int, ms to wait before trying next attempt with all waiting logic applied.
    """
    wait_time_ms = sleep_ms
    if wait_time_ms:
        if self._exponential_sleep_multiplier:
            hundred_years_ms = 100 * 365 * 86400 * 1000
            if self._exponential_sleep_multiplier > 1.0 and last_retrial > math.log(hundred_years_ms / wait_time_ms, self._exponential_sleep_multiplier):
                wait_time_ms = hundred_years_ms
            else:
                wait_time_ms *= self._exponential_sleep_multiplier ** last_retrial
        if self._jitter_ms:
            wait_time_ms += random.random() * self._jitter_ms
        if self._wait_ceiling_ms:
            wait_time_ms = min(wait_time_ms, self._wait_ceiling_ms)
        return wait_time_ms
    return 0