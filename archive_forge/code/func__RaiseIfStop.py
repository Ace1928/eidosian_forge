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
def _RaiseIfStop(self, result, state):
    if self._max_retrials is not None and self._max_retrials <= state.retrial:
        raise MaxRetrialsException('Reached', result, state)
    if self._max_wait_ms is not None:
        if state.time_passed_ms + state.time_to_wait_ms > self._max_wait_ms:
            raise WaitException('Timeout', result, state)