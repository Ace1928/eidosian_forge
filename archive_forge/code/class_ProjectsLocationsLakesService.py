from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsLakesService(base_api.BaseApiService):
    """Service class for the projects_locations_lakes resource."""
    _NAME = 'projects_locations_lakes'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsLakesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a lake resource.

      Args:
        request: (DataplexProjectsLocationsLakesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes', http_method='POST', method_id='dataplex.projects.locations.lakes.create', ordered_params=['parent'], path_params=['parent'], query_params=['lakeId', 'validateOnly'], relative_path='v1/{+parent}/lakes', request_field='googleCloudDataplexV1Lake', request_type_name='DataplexProjectsLocationsLakesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a lake resource. All zones within the lake must be deleted before the lake can be deleted.

      Args:
        request: (DataplexProjectsLocationsLakesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}', http_method='DELETE', method_id='dataplex.projects.locations.lakes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a lake resource.

      Args:
        request: (DataplexProjectsLocationsLakesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Lake) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}', http_method='GET', method_id='dataplex.projects.locations.lakes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesGetRequest', response_type_name='GoogleCloudDataplexV1Lake', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataplexProjectsLocationsLakesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}:getIamPolicy', http_method='GET', method_id='dataplex.projects.locations.lakes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='DataplexProjectsLocationsLakesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists lake resources in a project and location.

      Args:
        request: (DataplexProjectsLocationsLakesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListLakesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes', http_method='GET', method_id='dataplex.projects.locations.lakes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/lakes', request_field='', request_type_name='DataplexProjectsLocationsLakesListRequest', response_type_name='GoogleCloudDataplexV1ListLakesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a lake resource.

      Args:
        request: (DataplexProjectsLocationsLakesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}', http_method='PATCH', method_id='dataplex.projects.locations.lakes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudDataplexV1Lake', request_type_name='DataplexProjectsLocationsLakesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataplexProjectsLocationsLakesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}:setIamPolicy', http_method='POST', method_id='dataplex.projects.locations.lakes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='DataplexProjectsLocationsLakesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataplexProjectsLocationsLakesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}:testIamPermissions', http_method='POST', method_id='dataplex.projects.locations.lakes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='DataplexProjectsLocationsLakesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)