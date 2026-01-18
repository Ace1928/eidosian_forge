from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accessapproval.v1 import accessapproval_v1_messages as messages
def DeleteAccessApprovalSettings(self, request, global_params=None):
    """Deletes the settings associated with a project, folder, or organization. This will have the effect of disabling Access Approval for the project, folder, or organization, but only if all ancestors also have Access Approval disabled. If Access Approval is enabled at a higher level of the hierarchy, then Access Approval will still be enabled at this level as the settings are inherited.

      Args:
        request: (AccessapprovalProjectsDeleteAccessApprovalSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('DeleteAccessApprovalSettings')
    return self._RunMethod(config, request, global_params=global_params)