from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
class ProjectsLocationsCustomTargetTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_customTargetTypes resource."""
    _NAME = 'projects_locations_customTargetTypes'

    def __init__(self, client):
        super(ClouddeployV1.ProjectsLocationsCustomTargetTypesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new CustomTargetType in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsCustomTargetTypesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customTargetTypes', http_method='POST', method_id='clouddeploy.projects.locations.customTargetTypes.create', ordered_params=['parent'], path_params=['parent'], query_params=['customTargetTypeId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/customTargetTypes', request_field='customTargetType', request_type_name='ClouddeployProjectsLocationsCustomTargetTypesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single CustomTargetType.

      Args:
        request: (ClouddeployProjectsLocationsCustomTargetTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customTargetTypes/{customTargetTypesId}', http_method='DELETE', method_id='clouddeploy.projects.locations.customTargetTypes.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'requestId', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsCustomTargetTypesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single CustomTargetType.

      Args:
        request: (ClouddeployProjectsLocationsCustomTargetTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CustomTargetType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customTargetTypes/{customTargetTypesId}', http_method='GET', method_id='clouddeploy.projects.locations.customTargetTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsCustomTargetTypesGetRequest', response_type_name='CustomTargetType', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ClouddeployProjectsLocationsCustomTargetTypesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customTargetTypes/{customTargetTypesId}:getIamPolicy', http_method='GET', method_id='clouddeploy.projects.locations.customTargetTypes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ClouddeployProjectsLocationsCustomTargetTypesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CustomTargetTypes in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsCustomTargetTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCustomTargetTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customTargetTypes', http_method='GET', method_id='clouddeploy.projects.locations.customTargetTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/customTargetTypes', request_field='', request_type_name='ClouddeployProjectsLocationsCustomTargetTypesListRequest', response_type_name='ListCustomTargetTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single CustomTargetType.

      Args:
        request: (ClouddeployProjectsLocationsCustomTargetTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customTargetTypes/{customTargetTypesId}', http_method='PATCH', method_id='clouddeploy.projects.locations.customTargetTypes.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='customTargetType', request_type_name='ClouddeployProjectsLocationsCustomTargetTypesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ClouddeployProjectsLocationsCustomTargetTypesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customTargetTypes/{customTargetTypesId}:setIamPolicy', http_method='POST', method_id='clouddeploy.projects.locations.customTargetTypes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ClouddeployProjectsLocationsCustomTargetTypesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)