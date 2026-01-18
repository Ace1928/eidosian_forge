from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha import osconfig_v1alpha_messages as messages
class ProjectsLocationsInstanceOSPoliciesCompliancesService(base_api.BaseApiService):
    """Service class for the projects_locations_instanceOSPoliciesCompliances resource."""
    _NAME = 'projects_locations_instanceOSPoliciesCompliances'

    def __init__(self, client):
        super(OsconfigV1alpha.ProjectsLocationsInstanceOSPoliciesCompliancesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get OS policies compliance data for the specified Compute Engine VM instance.

      Args:
        request: (OsconfigProjectsLocationsInstanceOSPoliciesCompliancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceOSPoliciesCompliance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/instanceOSPoliciesCompliances/{instanceOSPoliciesCompliancesId}', http_method='GET', method_id='osconfig.projects.locations.instanceOSPoliciesCompliances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='OsconfigProjectsLocationsInstanceOSPoliciesCompliancesGetRequest', response_type_name='InstanceOSPoliciesCompliance', supports_download=False)

    def List(self, request, global_params=None):
        """List OS policies compliance data for all Compute Engine VM instances in the specified zone.

      Args:
        request: (OsconfigProjectsLocationsInstanceOSPoliciesCompliancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstanceOSPoliciesCompliancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/instanceOSPoliciesCompliances', http_method='GET', method_id='osconfig.projects.locations.instanceOSPoliciesCompliances.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/instanceOSPoliciesCompliances', request_field='', request_type_name='OsconfigProjectsLocationsInstanceOSPoliciesCompliancesListRequest', response_type_name='ListInstanceOSPoliciesCompliancesResponse', supports_download=False)