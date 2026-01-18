from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from apitools.base.py import exceptions as base_exceptions
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.core.console import progress_tracker as console_progress_tracker
from googlecloudsdk.core.util import retry
class OperationsV1Beta4(_BaseOperations):
    """Common utility functions for sql operations V1Beta4."""

    @staticmethod
    def GetOperation(sql_client, operation_ref, progress_tracker=None):
        """Helper function for getting the status of an operation for V1Beta4 API.

    Args:
      sql_client: apitools.BaseApiClient, The client used to make requests.
      operation_ref: resources.Resource, A reference for the operation to poll.
      progress_tracker: progress_tracker.ProgressTracker, A reference for the
          progress tracker to tick, in case this function is used in a Retryer.

    Returns:
      Operation: if the operation succeeded without error or  is not yet done.
      OperationError: If the operation has an error code or is in UNKNOWN state.
      Exception: Any other exception that can occur when calling Get
    """
        if progress_tracker:
            progress_tracker.Tick()
        try:
            op = sql_client.operations.Get(sql_client.MESSAGES_MODULE.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
        except Exception as e:
            return e
        if op.error and op.error.errors:
            error_object = op.error.errors[0]
            error = '[{}]'.format(error_object.code)
            if error_object.message:
                error += ' ' + error_object.message
            return exceptions.OperationError(error)
        if op.status == sql_client.MESSAGES_MODULE.Operation.StatusValueValuesEnum.SQL_OPERATION_STATUS_UNSPECIFIED:
            return exceptions.OperationError(op.status)
        return op

    @staticmethod
    def GetOperationWaitCommand(operation_ref):
        return 'gcloud beta sql operations wait --project {0} {1}'.format(operation_ref.project, operation_ref.operation)