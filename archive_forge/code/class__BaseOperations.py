from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from apitools.base.py import exceptions as base_exceptions
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.core.console import progress_tracker as console_progress_tracker
from googlecloudsdk.core.util import retry
class _BaseOperations(object):
    """Common utility functions for sql operations."""
    _PRE_START_SLEEP_SEC = 1
    _INITIAL_SLEEP_MS = 2000
    _WAIT_CEILING_MS = 20000
    _HTTP_MAX_RETRY_MS = 2000

    @classmethod
    def WaitForOperation(cls, sql_client, operation_ref, message, max_wait_seconds=_OPERATION_TIMEOUT_SECONDS):
        """Wait for a Cloud SQL operation to complete.

    No operation is done instantly. Wait for it to finish following this logic:
    First wait 1s, then query, then retry waiting exponentially more from 2s.
    We want to limit to 20s between retries to maintain some responsiveness.
    Finally, we want to limit the whole process to a conservative 300s. If we
    get to that point it means something is wrong and we can throw an exception.

    Args:
      sql_client: apitools.BaseApiClient, The client used to make requests.
      operation_ref: resources.Resource, A reference for the operation to poll.
      message: str, The string to print while polling.
      max_wait_seconds: integer or None, the number of seconds before the
          poller times out.

    Returns:
      Operation: The polled operation.

    Raises:
      OperationError: If the operation has an error code, is in UNKNOWN state,
          or if the operation takes more than max_wait_seconds when a value is
          specified.
    """

        def ShouldRetryFunc(result, state):
            if isinstance(result, base_exceptions.HttpError):
                if state.time_passed_ms > _BaseOperations._HTTP_MAX_RETRY_MS:
                    raise result
                return True
            if isinstance(result, Exception):
                raise result
            is_operation_done = result.status == sql_client.MESSAGES_MODULE.Operation.StatusValueValuesEnum.DONE
            return not is_operation_done
        max_wait_ms = None
        if max_wait_seconds:
            max_wait_ms = max_wait_seconds * _MS_PER_SECOND
        with console_progress_tracker.ProgressTracker(message, autotick=False) as pt:
            time.sleep(_BaseOperations._PRE_START_SLEEP_SEC)
            retryer = retry.Retryer(exponential_sleep_multiplier=2, max_wait_ms=max_wait_ms, wait_ceiling_ms=_BaseOperations._WAIT_CEILING_MS)
            try:
                return retryer.RetryOnResult(cls.GetOperation, [sql_client, operation_ref], {'progress_tracker': pt}, should_retry_if=ShouldRetryFunc, sleep_ms=_BaseOperations._INITIAL_SLEEP_MS)
            except retry.WaitException:
                raise exceptions.OperationError('Operation {0} is taking longer than expected. You can continue waiting for the operation by running `{1}`'.format(operation_ref, cls.GetOperationWaitCommand(operation_ref)))