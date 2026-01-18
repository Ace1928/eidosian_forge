from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
def ApproveRedaction(self, request, global_params=None):
    """Once the impact assessment completes, the redaction operation will move into WAIT_FOR_USER_APPROVAL stage wherein it's going to wait for the user to approve the redaction operation. Please note that the operation will be in progress at this point and if the user doesn't approve the redaction operation within the grace period, it will be auto-cancelled.The redaction operation can also be approved before operation moves into the WAIT_FOR_USER_APPROVAL stage. In that case redaction process will commence as soon as the impact assessment is complete. This is functionally similar to approving after the operation moves to WAIT_FOR_USER_APPROVAL stage but without any wait time to begin redaction.Once the user approves, the redaction operation will begin redacting the log entries.

      Args:
        request: (LoggingProjectsLocationsOperationsApproveRedactionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApproveRedactionOperationResponse) The response message.
      """
    config = self.GetMethodConfig('ApproveRedaction')
    return self._RunMethod(config, request, global_params=global_params)