from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.api_lib.dataproc.poller import (
from googlecloudsdk.core import log
class BatchPoller(dataproc_poller_base.AbstractOperationStreamerPoller):
    """Poller for batch workload."""

    def IsDone(self, batch):
        """See base class."""
        if batch and batch.state in (self.dataproc.messages.Batch.StateValueValuesEnum.SUCCEEDED, self.dataproc.messages.Batch.StateValueValuesEnum.CANCELLED, self.dataproc.messages.Batch.StateValueValuesEnum.FAILED):
            return True
        return False

    def Poll(self, batch_ref):
        """See base class."""
        request = self.dataproc.messages.DataprocProjectsLocationsBatchesGetRequest(name=batch_ref)
        try:
            return self.dataproc.client.projects_locations_batches.Get(request)
        except apitools_exceptions.HttpError as error:
            log.warning('Get Batch failed:\n{}'.format(error))
            if util.IsClientHttpException(error):
                raise

    def _GetResult(self, batch):
        """Handles errors.

    Error handling for batch jobs. This happen after the batch reaches one of
    the complete states.

    Overrides.

    Args:
      batch: The batch resource.

    Returns:
      None. The result is directly output to log.err.

    Raises:
      JobTimeoutError: When waiter timed out.
      JobError: When remote batch job is failed.
    """
        if not batch:
            raise exceptions.JobTimeoutError('Timed out while waiting for batch job.')
        if batch.state == self.dataproc.messages.Batch.StateValueValuesEnum.SUCCEEDED:
            if not self.driver_log_streamer:
                log.warning('Expected batch job output not found.')
            elif self.driver_log_streamer.open:
                log.warning('Batch job terminated, but output did not finish streaming.')
        elif batch.state == self.dataproc.messages.Batch.StateValueValuesEnum.CANCELLED:
            log.warning('Batch job is CANCELLED.')
        else:
            err_message = 'Batch job is FAILED.'
            if batch.stateMessage:
                err_message = '{} Detail: {}'.format(err_message, batch.stateMessage)
                if err_message[-1] != '.':
                    err_message += '.'
            err_message += '\n'
            err_message += "Running auto diagnostics on the batch. It may take few minutes before diagnostics output is available. Please check diagnostics output by running 'gcloud dataproc batches describe' command."
            raise exceptions.JobError(err_message)
        return None

    def _GetOutputUri(self, batch):
        """See base class."""
        if batch and batch.runtimeInfo and batch.runtimeInfo.outputUri:
            return batch.runtimeInfo.outputUri
        return None