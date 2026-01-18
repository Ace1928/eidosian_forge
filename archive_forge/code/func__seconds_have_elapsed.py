from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
def _seconds_have_elapsed(token, num_seconds):
    """Tests if 'num_seconds' have passed since 'token' was requested.

  Not strictly thread-safe - may log with the wrong frequency if called
  concurrently from multiple threads. Accuracy depends on resolution of
  'timeit.default_timer()'.

  Always returns True on the first call for a given 'token'.

  Args:
    token: The token for which to look up the count.
    num_seconds: The number of seconds to test for.

  Returns:
    Whether it has been >= 'num_seconds' since 'token' was last requested.
  """
    now = timeit.default_timer()
    then = _log_timer_per_token.get(token, None)
    if then is None or now - then >= num_seconds:
        _log_timer_per_token[token] = now
        return True
    else:
        return False