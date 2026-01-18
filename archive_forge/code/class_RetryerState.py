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
class RetryerState(object):
    """Object that holds the state of the retryer."""

    def __init__(self, retrial, time_passed_ms, time_to_wait_ms):
        """Initializer for RetryerState.

    Args:
      retrial: int, the retry attempt we are currently at.
      time_passed_ms: int, number of ms that passed since we started retryer.
      time_to_wait_ms: int, number of ms to wait for the until next trial.
          If this number is -1, it means the iterable item that specifies the
          next sleep value has raised StopIteration.
    """
        self.retrial = retrial
        self.time_passed_ms = time_passed_ms
        self.time_to_wait_ms = time_to_wait_ms