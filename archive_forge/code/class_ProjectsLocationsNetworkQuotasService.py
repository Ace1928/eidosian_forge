from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsNetworkQuotasService(base_api.BaseApiService):
    """Service class for the projects_locations_networkQuotas resource."""
    _NAME = 'projects_locations_networkQuotas'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsNetworkQuotasService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List Network provisioning quotas.

      Args:
        request: (BaremetalsolutionProjectsLocationsNetworkQuotasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNetworkQuotasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/networkQuotas', http_method='GET', method_id='baremetalsolution.projects.locations.networkQuotas.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/networkQuotas', request_field='', request_type_name='BaremetalsolutionProjectsLocationsNetworkQuotasListRequest', response_type_name='ListNetworkQuotasResponse', supports_download=False)