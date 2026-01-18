from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsLoggingServersService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_loggingServers resource."""
    _NAME = 'projects_locations_privateClouds_loggingServers'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsLoggingServersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new logging server for a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsLoggingServersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/loggingServers', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.loggingServers.create', ordered_params=['parent'], path_params=['parent'], query_params=['loggingServerId', 'requestId'], relative_path='v1/{+parent}/loggingServers', request_field='loggingServer', request_type_name='VmwareengineProjectsLocationsPrivateCloudsLoggingServersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single logging server.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsLoggingServersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/loggingServers/{loggingServersId}', http_method='DELETE', method_id='vmwareengine.projects.locations.privateClouds.loggingServers.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsLoggingServersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a logging server.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsLoggingServersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LoggingServer) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/loggingServers/{loggingServersId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.loggingServers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsLoggingServersGetRequest', response_type_name='LoggingServer', supports_download=False)

    def List(self, request, global_params=None):
        """Lists logging servers configured for a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsLoggingServersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLoggingServersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/loggingServers', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.loggingServers.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/loggingServers', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsLoggingServersListRequest', response_type_name='ListLoggingServersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single logging server. Only fields specified in `update_mask` are applied.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsLoggingServersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/loggingServers/{loggingServersId}', http_method='PATCH', method_id='vmwareengine.projects.locations.privateClouds.loggingServers.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='loggingServer', request_type_name='VmwareengineProjectsLocationsPrivateCloudsLoggingServersPatchRequest', response_type_name='Operation', supports_download=False)