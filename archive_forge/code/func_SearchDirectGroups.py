from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def SearchDirectGroups(self, request, global_params=None):
    """Searches direct groups of a member.

      Args:
        request: (CloudidentityGroupsMembershipsSearchDirectGroupsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchDirectGroupsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchDirectGroups')
    return self._RunMethod(config, request, global_params=global_params)