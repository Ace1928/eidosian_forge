from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import enum
from googlecloudsdk.api_lib.app import exceptions as app_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
class AppEngineOperationPoller(waiter.OperationPoller):
    """A poller for appengine operations."""

    def __init__(self, operation_service, operation_metadata_type=None):
        """Sets up poller for appengine operations.

    Args:
      operation_service: apitools.base.py.base_api.BaseApiService, api service
        for retrieving information about ongoing operation.
      operation_metadata_type: Message class for the Operation metadata (for
        instance, OperationMetadataV1, or OperationMetadataV1Beta).
    """
        self.operation_service = operation_service
        self.operation_metadata_type = operation_metadata_type
        self.warnings_seen = set()

    def IsDone(self, operation):
        """Overrides."""
        self._LogNewWarnings(operation)
        if operation.done:
            log.debug('Operation [{0}] complete. Result: {1}'.format(operation.name, json.dumps(encoding.MessageToDict(operation), indent=4)))
            if operation.error:
                raise OperationError(requests.ExtractErrorMessage(encoding.MessageToPyValue(operation.error)))
            return True
        log.debug('Operation [{0}] not complete. Waiting to retry.'.format(operation.name))
        return False

    def Poll(self, operation_ref):
        """Overrides.

    Args:
      operation_ref: googlecloudsdk.core.resources.Resource.

    Returns:
      fetched operation message.
    """
        request_type = self.operation_service.GetRequestType('Get')
        request = request_type(name=operation_ref.RelativeName())
        operation = self.operation_service.Get(request)
        self._LogNewWarnings(operation)
        return operation

    def _LogNewWarnings(self, operation):
        if self.operation_metadata_type:
            new_warnings = GetWarningsFromOperation(operation, self.operation_metadata_type) - self.warnings_seen
            for warning in new_warnings:
                log.warning(warning + '\n')
                self.warnings_seen.add(warning)

    def GetResult(self, operation):
        """Simply returns the operation.

    Args:
      operation: api_name_messages.Operation.

    Returns:
      the 'response' field of the Operation.
    """
        return operation