from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def PollUntilDone(poller, operation_ref, max_retrials=None, max_wait_ms=1800000, exponential_sleep_multiplier=1.4, jitter_ms=1000, wait_ceiling_ms=180000, sleep_ms=2000, status_update=None):
    """Waits for poller.Poll to complete.

  Note that this *does not* print nice messages to stderr for the user; most
  callers should use WaitFor instead for the best UX unless there's a good
  reason not to print.

  Args:
    poller: OperationPoller, poller to use during retrials.
    operation_ref: object, passed to operation poller poll method.
    max_retrials: int, max number of retrials before raising RetryException.
    max_wait_ms: int, number of ms to wait before raising WaitException.
    exponential_sleep_multiplier: float, factor to use on subsequent retries.
    jitter_ms: int, random (up to the value) additional sleep between retries.
    wait_ceiling_ms: int, Maximum wait between retries.
    sleep_ms: int or iterable: for how long to wait between trials.
    status_update: func(result, state) called right after each trial.

  Returns:
    The return value from poller.Poll.
  """
    retryer = retry.Retryer(max_retrials=max_retrials, max_wait_ms=max_wait_ms, exponential_sleep_multiplier=exponential_sleep_multiplier, jitter_ms=jitter_ms, wait_ceiling_ms=wait_ceiling_ms, status_update_func=status_update)

    def _IsNotDone(operation, unused_state):
        return not poller.IsDone(operation)
    operation = retryer.RetryOnResult(func=poller.Poll, args=(operation_ref,), should_retry_if=_IsNotDone, sleep_ms=sleep_ms)
    return operation