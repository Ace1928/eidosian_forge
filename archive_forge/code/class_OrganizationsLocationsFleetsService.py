from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class OrganizationsLocationsFleetsService(base_api.BaseApiService):
    """Service class for the organizations_locations_fleets resource."""
    _NAME = 'organizations_locations_fleets'

    def __init__(self, client):
        super(GkehubV1beta.OrganizationsLocationsFleetsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns all fleets within an organization or a project that the caller has access to.

      Args:
        request: (GkehubOrganizationsLocationsFleetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFleetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/fleets', http_method='GET', method_id='gkehub.organizations.locations.fleets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/fleets', request_field='', request_type_name='GkehubOrganizationsLocationsFleetsListRequest', response_type_name='ListFleetsResponse', supports_download=False)