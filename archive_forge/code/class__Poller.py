from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
class _Poller(waiter.CloudOperationPollerNoResources):
    """Poller for Anthos Multi-cloud operations.

  The poller stores the status detail from the operation message to update
  the progress tracker.
  """

    def __init__(self, operation_service):
        """See base class."""
        self.operation_service = operation_service
        self.status_detail = None
        self.last_error_detail = None

    def Poll(self, operation_ref):
        """See base class."""
        request_type = self.operation_service.GetRequestType('Get')
        operation = self.operation_service.Get(request_type(name=operation_ref.RelativeName()))
        if operation.metadata is not None:
            metadata = encoding.MessageToPyValue(operation.metadata)
            if 'statusDetail' in metadata:
                self.status_detail = metadata['statusDetail']
            if 'errorDetail' in metadata:
                error_detail = metadata['errorDetail']
                if error_detail != self.last_error_detail:
                    log.error(error_detail)
                self.last_error_detail = error_detail
        return operation

    def GetDetailMessage(self):
        return self.status_detail