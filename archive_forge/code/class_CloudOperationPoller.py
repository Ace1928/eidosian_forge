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
class CloudOperationPoller(OperationPoller):
    """Manages a longrunning Operations.

  See https://cloud.google.com/speech/reference/rpc/google.longrunning
  """

    def __init__(self, result_service, operation_service):
        """Sets up poller for cloud operations.

    Args:
      result_service: apitools.base.py.base_api.BaseApiService, api service for
        retrieving created result of initiated operation.
      operation_service: apitools.base.py.base_api.BaseApiService, api service
        for retrieving information about ongoing operation.

      Note that result_service and operation_service Get request must have
      single attribute called 'name'.
    """
        self.result_service = result_service
        self.operation_service = operation_service

    def IsDone(self, operation):
        """Overrides."""
        if operation.done:
            if operation.error:
                raise OperationError(operation.error.message)
            return True
        return False

    def Poll(self, operation_ref):
        """Overrides.

    Args:
      operation_ref: googlecloudsdk.core.resources.Resource.

    Returns:
      fetched operation message.
    """
        request_type = self.operation_service.GetRequestType('Get')
        return self.operation_service.Get(request_type(name=operation_ref.RelativeName()))

    def GetResult(self, operation):
        """Overrides.

    Args:
      operation: api_name_messages.Operation.

    Returns:
      result of result_service.Get request.
    """
        request_type = self.result_service.GetRequestType('Get')
        response_dict = encoding.MessageToPyValue(operation.response)
        return self.result_service.Get(request_type(name=response_dict['name']))