from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def CheckTransitiveMembership(self, request, global_params=None):
    """Check a potential member for membership in a group. **Note:** This feature is only available to Google Workspace Enterprise Standard, Enterprise Plus, and Enterprise for Education; and Cloud Identity Premium accounts. If the account of the member is not one of these, a 403 (PERMISSION_DENIED) HTTP status code will be returned. A member has membership to a group as long as there is a single viewable transitive membership between the group and the member. The actor must have view permissions to at least one transitive membership between the member and group.

      Args:
        request: (CloudidentityGroupsMembershipsCheckTransitiveMembershipRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckTransitiveMembershipResponse) The response message.
      """
    config = self.GetMethodConfig('CheckTransitiveMembership')
    return self._RunMethod(config, request, global_params=global_params)