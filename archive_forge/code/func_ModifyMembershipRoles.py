from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def ModifyMembershipRoles(self, request, global_params=None):
    """Modifies the `MembershipRole`s of a `Membership`.

      Args:
        request: (CloudidentityGroupsMembershipsModifyMembershipRolesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModifyMembershipRolesResponse) The response message.
      """
    config = self.GetMethodConfig('ModifyMembershipRoles')
    return self._RunMethod(config, request, global_params=global_params)