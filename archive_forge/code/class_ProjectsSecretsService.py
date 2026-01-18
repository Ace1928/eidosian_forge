from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.secretmanager.v1 import secretmanager_v1_messages as messages
class ProjectsSecretsService(base_api.BaseApiService):
    """Service class for the projects_secrets resource."""
    _NAME = 'projects_secrets'

    def __init__(self, client):
        super(SecretmanagerV1.ProjectsSecretsService, self).__init__(client)
        self._upload_configs = {}

    def AddVersion(self, request, global_params=None):
        """Creates a new SecretVersion containing secret data and attaches it to an existing Secret.

      Args:
        request: (SecretmanagerProjectsSecretsAddVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecretVersion) The response message.
      """
        config = self.GetMethodConfig('AddVersion')
        return self._RunMethod(config, request, global_params=global_params)
    AddVersion.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}:addVersion', http_method='POST', method_id='secretmanager.projects.secrets.addVersion', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:addVersion', request_field='addSecretVersionRequest', request_type_name='SecretmanagerProjectsSecretsAddVersionRequest', response_type_name='SecretVersion', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Secret containing no SecretVersions.

      Args:
        request: (SecretmanagerProjectsSecretsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets', http_method='POST', method_id='secretmanager.projects.secrets.create', ordered_params=['parent'], path_params=['parent'], query_params=['secretId'], relative_path='v1/{+parent}/secrets', request_field='secret', request_type_name='SecretmanagerProjectsSecretsCreateRequest', response_type_name='Secret', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Secret.

      Args:
        request: (SecretmanagerProjectsSecretsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}', http_method='DELETE', method_id='secretmanager.projects.secrets.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='SecretmanagerProjectsSecretsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets metadata for a given Secret.

      Args:
        request: (SecretmanagerProjectsSecretsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}', http_method='GET', method_id='secretmanager.projects.secrets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecretmanagerProjectsSecretsGetRequest', response_type_name='Secret', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a secret. Returns empty policy if the secret exists and does not have a policy set.

      Args:
        request: (SecretmanagerProjectsSecretsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}:getIamPolicy', http_method='GET', method_id='secretmanager.projects.secrets.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='SecretmanagerProjectsSecretsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Secrets.

      Args:
        request: (SecretmanagerProjectsSecretsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSecretsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets', http_method='GET', method_id='secretmanager.projects.secrets.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/secrets', request_field='', request_type_name='SecretmanagerProjectsSecretsListRequest', response_type_name='ListSecretsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates metadata of an existing Secret.

      Args:
        request: (SecretmanagerProjectsSecretsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}', http_method='PATCH', method_id='secretmanager.projects.secrets.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='secret', request_type_name='SecretmanagerProjectsSecretsPatchRequest', response_type_name='Secret', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified secret. Replaces any existing policy. Permissions on SecretVersions are enforced according to the policy set on the associated Secret.

      Args:
        request: (SecretmanagerProjectsSecretsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}:setIamPolicy', http_method='POST', method_id='secretmanager.projects.secrets.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='SecretmanagerProjectsSecretsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has for the specified secret. If the secret does not exist, this call returns an empty set of permissions, not a NOT_FOUND error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (SecretmanagerProjectsSecretsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}:testIamPermissions', http_method='POST', method_id='secretmanager.projects.secrets.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='SecretmanagerProjectsSecretsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)