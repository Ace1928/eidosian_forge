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
def WaitFor(poller, operation_ref, message=None, custom_tracker=None, tracker_update_func=None, pre_start_sleep_ms=1000, max_retrials=None, max_wait_ms=1800000, exponential_sleep_multiplier=1.4, jitter_ms=1000, wait_ceiling_ms=180000, sleep_ms=2000):
    """Waits for poller.Poll and displays pending operation spinner.

  Args:
    poller: OperationPoller, poller to use during retrials.
    operation_ref: object, passed to operation poller poll method.
    message: str, string to display for default progress_tracker.
    custom_tracker: ProgressTracker, progress_tracker to use for display.
    tracker_update_func: func(tracker, result, status), tracker update function.
    pre_start_sleep_ms: int, Time to wait before making first poll request.
    max_retrials: int, max number of retrials before raising RetryException.
    max_wait_ms: int, number of ms to wait before raising WaitException.
    exponential_sleep_multiplier: float, factor to use on subsequent retries.
    jitter_ms: int, random (up to the value) additional sleep between retries.
    wait_ceiling_ms: int, Maximum wait between retries.
    sleep_ms: int or iterable: for how long to wait between trials.

  Returns:
    poller.GetResult(operation).

  Raises:
    AbortWaitError: if ctrl-c was pressed.
    TimeoutError: if retryer has finished without being done.
  """
    aborted_message = 'Aborting wait for operation {0}.\n'.format(operation_ref)
    try:
        with progress_tracker.ProgressTracker(message, aborted_message=aborted_message) if not custom_tracker else custom_tracker as tracker:
            if pre_start_sleep_ms:
                _SleepMs(pre_start_sleep_ms)

            def _StatusUpdate(result, status):
                if tracker_update_func:
                    tracker_update_func(tracker, result, status)
                else:
                    tracker.Tick()
            operation = PollUntilDone(poller, operation_ref, max_retrials, max_wait_ms, exponential_sleep_multiplier, jitter_ms, wait_ceiling_ms, sleep_ms, _StatusUpdate)
    except retry.WaitException:
        raise TimeoutError('Operation {0} has not finished in {1} seconds. {2}'.format(operation_ref, max_wait_ms // 1000, _TIMEOUT_MESSAGE))
    except retry.MaxRetrialsException as e:
        raise TimeoutError('Operation {0} has not finished in {1} seconds after max {2} retrials. {3}'.format(operation_ref, e.state.time_passed_ms // 1000, e.state.retrial, _TIMEOUT_MESSAGE))
    return poller.GetResult(operation)