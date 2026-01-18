from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
def RetryNoBackoff(callable_func, retry_notify_func, delay=5, max_tries=200):
    """Calls a function multiple times, with the same delay each time.

  Args:
    callable_func: A function that performs some operation that should be
      retried a number of times upon failure.  Signature: () -> (done, value)
      If 'done' is True, we'll immediately return (True, value)
      If 'done' is False, we'll delay a bit and try again, unless we've
      hit the 'max_tries' limit, in which case we'll return (False, value).
    retry_notify_func: This function will be called immediately before the
      next retry delay.  Signature: (value, delay) -> None
      'value' is the value returned by the last call to 'callable_func'
      'delay' is the retry delay, in seconds
    delay: Delay between tries, in seconds.
    max_tries: Maximum number of tries (the first one counts).

  Returns:
    What the last call to 'callable_func' returned, which is of the form
    (done, value).  If 'done' is True, you know 'callable_func' returned True
    before we ran out of retries.  If 'done' is False, you know 'callable_func'
    kept returning False and we ran out of retries.

  Raises:
    Whatever the function raises--an exception will immediately stop retries.
  """
    return RetryWithBackoff(callable_func, retry_notify_func, delay, 1, delay, max_tries)