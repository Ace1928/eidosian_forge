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
class RetryException(Exception):
    """Raised to stop retrials on failure."""

    def __init__(self, message, last_result, state):
        self.message = message
        self.last_result = last_result
        self.state = state
        super(RetryException, self).__init__(message)

    def __str__(self):
        return 'last_result={last_result}, last_retrial={last_retrial}, time_passed_ms={time_passed_ms},time_to_wait={time_to_wait_ms}'.format(last_result=self.last_result, last_retrial=self.state.retrial, time_passed_ms=self.state.time_passed_ms, time_to_wait_ms=self.state.time_to_wait_ms)