from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsLocationsWorkloadIdentityPoolsProvidersKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadIdentityPools_providers_keys resource."""
    _NAME = 'projects_locations_workloadIdentityPools_providers_keys'

    def __init__(self, client):
        super(IamV1.ProjectsLocationsWorkloadIdentityPoolsProvidersKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new WorkloadIdentityPoolProviderKey in a WorkloadIdentityPoolProvider.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}/keys', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.providers.keys.create', ordered_params=['parent'], path_params=['parent'], query_params=['workloadIdentityPoolProviderKeyId'], relative_path='v1/{+parent}/keys', request_field='workloadIdentityPoolProviderKey', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an WorkloadIdentityPoolProviderKey. You can undelete a key for 30 days. After 30 days, deletion is permanent.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}/keys/{keysId}', http_method='DELETE', method_id='iam.projects.locations.workloadIdentityPools.providers.keys.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkloadIdentityPoolProviderKey.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkloadIdentityPoolProviderKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}/keys/{keysId}', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.providers.keys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysGetRequest', response_type_name='WorkloadIdentityPoolProviderKey', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkloadIdentityPoolProviderKeys in a project. If show_deleted is set to `true`, then deleted pools are also listed.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadIdentityPoolProviderKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}/keys', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.providers.keys.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/keys', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysListRequest', response_type_name='ListWorkloadIdentityPoolProviderKeysResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes an WorkloadIdentityPoolProviderKey, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}/keys/{keysId}:undelete', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.providers.keys.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkloadIdentityPoolProviderKeyRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysUndeleteRequest', response_type_name='Operation', supports_download=False)