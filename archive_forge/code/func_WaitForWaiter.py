from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import socket
from apitools.base.py import encoding
from googlecloudsdk.api_lib.runtime_config import exceptions as rtc_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as sdk_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def WaitForWaiter(waiter_resource, sleep=None, max_wait=None):
    """Wait for a waiter to finish.

  Args:
    waiter_resource: The waiter resource to wait for.
    sleep: The number of seconds to sleep between status checks.
    max_wait: The maximum number of seconds to wait before an error is raised.

  Returns:
    The last retrieved value of the Waiter.

  Raises:
    WaitTimeoutError: If the wait operation takes longer than the maximum wait
        time.
  """
    sleep = sleep if sleep is not None else DEFAULT_WAITER_SLEEP
    max_wait = max_wait if max_wait is not None else MAX_WAITER_TIMEOUT
    waiter_client = WaiterClient()
    retryer = retry.Retryer(max_wait_ms=max_wait * 1000)
    request = waiter_client.client.MESSAGES_MODULE.RuntimeconfigProjectsConfigsWaitersGetRequest(name=waiter_resource.RelativeName())
    with progress_tracker.ProgressTracker('Waiting for waiter [{0}] to finish'.format(waiter_resource.Name())):
        try:
            result = retryer.RetryOnResult(waiter_client.Get, args=[request], sleep_ms=sleep * 1000, should_retry_if=lambda w, s: not w.done)
        except retry.WaitException:
            raise rtc_exceptions.WaitTimeoutError('Waiter [{0}] did not finish within {1} seconds.'.format(waiter_resource.Name(), max_wait))
    if result.error is not None:
        if result.error.message is not None:
            message = 'Waiter [{0}] finished with an error: {1}'.format(waiter_resource.Name(), result.error.message)
        else:
            message = 'Waiter [{0}] finished with an error.'.format(waiter_resource.Name())
        log.error(message)
    return result