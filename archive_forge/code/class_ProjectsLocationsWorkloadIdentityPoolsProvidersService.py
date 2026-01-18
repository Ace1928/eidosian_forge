from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1beta import iam_v1beta_messages as messages
class ProjectsLocationsWorkloadIdentityPoolsProvidersService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadIdentityPools_providers resource."""
    _NAME = 'projects_locations_workloadIdentityPools_providers'

    def __init__(self, client):
        super(IamV1beta.ProjectsLocationsWorkloadIdentityPoolsProvidersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkloadIdentityPoolProvider in a WorkloadIdentityPool. You cannot reuse the name of a deleted provider until 30 days after deletion.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.providers.create', ordered_params=['parent'], path_params=['parent'], query_params=['workloadIdentityPoolProviderId'], relative_path='v1beta/{+parent}/providers', request_field='googleIamV1betaWorkloadIdentityPoolProvider', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkloadIdentityPoolProvider. Deleting a provider does not revoke credentials that have already been issued; they continue to grant access. You can undelete a provider for 30 days. After 30 days, deletion is permanent. You cannot update deleted providers. However, you can view and list them.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}', http_method='DELETE', method_id='iam.projects.locations.workloadIdentityPools.providers.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkloadIdentityPoolProvider.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1betaWorkloadIdentityPoolProvider) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.providers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersGetRequest', response_type_name='GoogleIamV1betaWorkloadIdentityPoolProvider', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkloadIdentityPoolProviders in a WorkloadIdentityPool. If `show_deleted` is set to `true`, then deleted providers are also listed.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1betaListWorkloadIdentityPoolProvidersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers', http_method='GET', method_id='iam.projects.locations.workloadIdentityPools.providers.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1beta/{+parent}/providers', request_field='', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersListRequest', response_type_name='GoogleIamV1betaListWorkloadIdentityPoolProvidersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkloadIdentityPoolProvider.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}', http_method='PATCH', method_id='iam.projects.locations.workloadIdentityPools.providers.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='googleIamV1betaWorkloadIdentityPoolProvider', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkloadIdentityPoolProvider, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsProvidersUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workloadIdentityPools/{workloadIdentityPoolsId}/providers/{providersId}:undelete', http_method='POST', method_id='iam.projects.locations.workloadIdentityPools.providers.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:undelete', request_field='googleIamV1betaUndeleteWorkloadIdentityPoolProviderRequest', request_type_name='IamProjectsLocationsWorkloadIdentityPoolsProvidersUndeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)