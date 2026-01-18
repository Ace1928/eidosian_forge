from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsSubnetsService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_subnets resource."""
    _NAME = 'projects_locations_privateClouds_subnets'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsSubnetsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single subnet.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsSubnetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subnet) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/subnets/{subnetsId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.subnets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsSubnetsGetRequest', response_type_name='Subnet', supports_download=False)

    def List(self, request, global_params=None):
        """Lists subnets in a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsSubnetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSubnetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/subnets', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.subnets.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/subnets', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsSubnetsListRequest', response_type_name='ListSubnetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single subnet. Only fields specified in `update_mask` are applied. *Note*: This API is synchronous and always returns a successful `google.longrunning.Operation` (LRO). The returned LRO will only have `done` and `response` fields.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsSubnetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/subnets/{subnetsId}', http_method='PATCH', method_id='vmwareengine.projects.locations.privateClouds.subnets.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='subnet', request_type_name='VmwareengineProjectsLocationsPrivateCloudsSubnetsPatchRequest', response_type_name='Operation', supports_download=False)