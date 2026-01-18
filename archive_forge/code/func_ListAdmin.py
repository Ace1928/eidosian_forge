from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def ListAdmin(self, request, global_params=None):
    """Lists Memberships of admin clusters in a given project and location. **This method is only used internally**.

      Args:
        request: (GkehubProjectsLocationsMembershipsListAdminRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAdminClusterMembershipsResponse) The response message.
      """
    config = self.GetMethodConfig('ListAdmin')
    return self._RunMethod(config, request, global_params=global_params)