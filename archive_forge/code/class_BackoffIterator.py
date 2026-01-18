import functools
import random
import time
class BackoffIterator:
    """iterate sleep value based on the exponential or jitter back-off algorithm.
    Args:
        delay (int or float): initial delay.
        backoff (int or float): backoff multiplier e.g. value of 2 will  double the delay each retry.
        max_delay (int or None): maximum amount of time to wait between retries.
        jitter (bool): if set to true, add jitter to the generate value.
    """

    def __init__(self, delay, backoff, max_delay=None, jitter=False):
        self.delay = delay
        self.backoff = backoff
        self.max_delay = max_delay
        self.jitter = jitter

    def __iter__(self):
        self.current_delay = self.delay
        return self

    def __next__(self):
        return_value = self.current_delay if self.max_delay is None else min(self.current_delay, self.max_delay)
        if self.jitter:
            return_value = random.uniform(0.0, return_value)
        self.current_delay *= self.backoff
        return return_value