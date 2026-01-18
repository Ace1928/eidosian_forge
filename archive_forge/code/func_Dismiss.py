from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accessapproval.v1 import accessapproval_v1_messages as messages
def Dismiss(self, request, global_params=None):
    """Dismisses a request. Returns the updated ApprovalRequest. NOTE: This does not deny access to the resource if another request has been made and approved. It is equivalent in effect to ignoring the request altogether. Returns NOT_FOUND if the request does not exist. Returns FAILED_PRECONDITION if the request exists but is not in a pending state.

      Args:
        request: (AccessapprovalProjectsApprovalRequestsDismissRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApprovalRequest) The response message.
      """
    config = self.GetMethodConfig('Dismiss')
    return self._RunMethod(config, request, global_params=global_params)