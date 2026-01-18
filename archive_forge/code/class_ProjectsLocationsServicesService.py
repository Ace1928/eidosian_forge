from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
class ProjectsLocationsServicesService(base_api.BaseApiService):
    """Service class for the projects_locations_services resource."""
    _NAME = 'projects_locations_services'

    def __init__(self, client):
        super(RunV2.ProjectsLocationsServicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Service in a given project and location.

      Args:
        request: (RunProjectsLocationsServicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services', http_method='POST', method_id='run.projects.locations.services.create', ordered_params=['parent'], path_params=['parent'], query_params=['serviceId', 'validateOnly'], relative_path='v2/{+parent}/services', request_field='googleCloudRunV2Service', request_type_name='RunProjectsLocationsServicesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Service. This will cause the Service to stop serving traffic and will delete all revisions.

      Args:
        request: (RunProjectsLocationsServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}', http_method='DELETE', method_id='run.projects.locations.services.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsServicesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a Service.

      Args:
        request: (RunProjectsLocationsServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2Service) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}', http_method='GET', method_id='run.projects.locations.services.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsServicesGetRequest', response_type_name='GoogleCloudRunV2Service', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM Access Control policy currently in effect for the given Cloud Run Service. This result does not include any inherited policies.

      Args:
        request: (RunProjectsLocationsServicesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}:getIamPolicy', http_method='GET', method_id='run.projects.locations.services.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v2/{+resource}:getIamPolicy', request_field='', request_type_name='RunProjectsLocationsServicesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Services.

      Args:
        request: (RunProjectsLocationsServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ListServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services', http_method='GET', method_id='run.projects.locations.services.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/services', request_field='', request_type_name='RunProjectsLocationsServicesListRequest', response_type_name='GoogleCloudRunV2ListServicesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Service.

      Args:
        request: (RunProjectsLocationsServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}', http_method='PATCH', method_id='run.projects.locations.services.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v2/{+name}', request_field='googleCloudRunV2Service', request_type_name='RunProjectsLocationsServicesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM Access control policy for the specified Service. Overwrites any existing policy.

      Args:
        request: (RunProjectsLocationsServicesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}:setIamPolicy', http_method='POST', method_id='run.projects.locations.services.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='RunProjectsLocationsServicesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified Project. There are no permissions required for making this API call.

      Args:
        request: (RunProjectsLocationsServicesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}:testIamPermissions', http_method='POST', method_id='run.projects.locations.services.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='RunProjectsLocationsServicesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)