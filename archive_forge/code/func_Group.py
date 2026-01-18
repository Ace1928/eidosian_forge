from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
def Group(self, request, global_params=None):
    """Filters an organization or source's findings and groups them by their specified properties in a location. If no location is specified, findings are assumed to be in global To group across all sources provide a `-` as the source id. The following list shows some examples: + `/v2/organizations/{organization_id}/sources/-/findings` + `/v2/organizations/{organization_id}/sources/-/locations/{location_id}/findings` + `/v2/folders/{folder_id}/sources/-/findings` + `/v2/folders/{folder_id}/sources/-/locations/{location_id}/findings` + `/v2/projects/{project_id}/sources/-/findings` + `/v2/projects/{project_id}/sources/-/locations/{location_id}/findings`.

      Args:
        request: (SecuritycenterProjectsSourcesLocationsFindingsGroupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GroupFindingsResponse) The response message.
      """
    config = self.GetMethodConfig('Group')
    return self._RunMethod(config, request, global_params=global_params)