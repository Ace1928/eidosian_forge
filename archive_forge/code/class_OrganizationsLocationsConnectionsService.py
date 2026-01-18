from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class OrganizationsLocationsConnectionsService(base_api.BaseApiService):
    """Service class for the organizations_locations_connections resource."""
    _NAME = 'organizations_locations_connections'

    def __init__(self, client):
        super(DlpV2.OrganizationsLocationsConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Search(self, request, global_params=None):
        """Searches for Connections in a parent.

      Args:
        request: (DlpOrganizationsLocationsConnectionsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2SearchConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/connections:search', http_method='GET', method_id='dlp.organizations.locations.connections.search', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/connections:search', request_field='', request_type_name='DlpOrganizationsLocationsConnectionsSearchRequest', response_type_name='GooglePrivacyDlpV2SearchConnectionsResponse', supports_download=False)