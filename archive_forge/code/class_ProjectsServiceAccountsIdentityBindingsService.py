from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsServiceAccountsIdentityBindingsService(base_api.BaseApiService):
    """Service class for the projects_serviceAccounts_identityBindings resource."""
    _NAME = 'projects_serviceAccounts_identityBindings'

    def __init__(self, client):
        super(IamV1.ProjectsServiceAccountsIdentityBindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create method for the projects_serviceAccounts_identityBindings service.

      Args:
        request: (IamProjectsServiceAccountsIdentityBindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccountIdentityBinding) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/identityBindings', http_method='POST', method_id='iam.projects.serviceAccounts.identityBindings.create', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/identityBindings', request_field='createServiceAccountIdentityBindingRequest', request_type_name='IamProjectsServiceAccountsIdentityBindingsCreateRequest', response_type_name='ServiceAccountIdentityBinding', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete method for the projects_serviceAccounts_identityBindings service.

      Args:
        request: (IamProjectsServiceAccountsIdentityBindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/identityBindings/{identityBindingsId}', http_method='DELETE', method_id='iam.projects.serviceAccounts.identityBindings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsServiceAccountsIdentityBindingsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get method for the projects_serviceAccounts_identityBindings service.

      Args:
        request: (IamProjectsServiceAccountsIdentityBindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccountIdentityBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/identityBindings/{identityBindingsId}', http_method='GET', method_id='iam.projects.serviceAccounts.identityBindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsServiceAccountsIdentityBindingsGetRequest', response_type_name='ServiceAccountIdentityBinding', supports_download=False)

    def List(self, request, global_params=None):
        """List method for the projects_serviceAccounts_identityBindings service.

      Args:
        request: (IamProjectsServiceAccountsIdentityBindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceAccountIdentityBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/identityBindings', http_method='GET', method_id='iam.projects.serviceAccounts.identityBindings.list', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/identityBindings', request_field='', request_type_name='IamProjectsServiceAccountsIdentityBindingsListRequest', response_type_name='ListServiceAccountIdentityBindingsResponse', supports_download=False)