from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1 import binaryauthorization_v1_messages as messages
class ProjectsPlatformsPoliciesService(base_api.BaseApiService):
    """Service class for the projects_platforms_policies resource."""
    _NAME = 'projects_platforms_policies'

    def __init__(self, client):
        super(BinaryauthorizationV1.ProjectsPlatformsPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a platform policy, and returns a copy of it. Returns `NOT_FOUND` if the project or platform doesn't exist, `INVALID_ARGUMENT` if the request is malformed, `ALREADY_EXISTS` if the policy already exists, and `INVALID_ARGUMENT` if the policy contains a platform-specific policy that does not match the platform value specified in the URL.

      Args:
        request: (BinaryauthorizationProjectsPlatformsPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PlatformPolicy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/platforms/{platformsId}/policies', http_method='POST', method_id='binaryauthorization.projects.platforms.policies.create', ordered_params=['parent'], path_params=['parent'], query_params=['policyId'], relative_path='v1/{+parent}/policies', request_field='platformPolicy', request_type_name='BinaryauthorizationProjectsPlatformsPoliciesCreateRequest', response_type_name='PlatformPolicy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a platform policy. Returns `NOT_FOUND` if the policy doesn't exist.

      Args:
        request: (BinaryauthorizationProjectsPlatformsPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/platforms/{platformsId}/policies/{policiesId}', http_method='DELETE', method_id='binaryauthorization.projects.platforms.policies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BinaryauthorizationProjectsPlatformsPoliciesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a platform policy. Returns `NOT_FOUND` if the policy doesn't exist.

      Args:
        request: (BinaryauthorizationProjectsPlatformsPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PlatformPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/platforms/{platformsId}/policies/{policiesId}', http_method='GET', method_id='binaryauthorization.projects.platforms.policies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BinaryauthorizationProjectsPlatformsPoliciesGetRequest', response_type_name='PlatformPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists platform policies owned by a project in the specified platform. Returns `INVALID_ARGUMENT` if the project or the platform doesn't exist.

      Args:
        request: (BinaryauthorizationProjectsPlatformsPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPlatformPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/platforms/{platformsId}/policies', http_method='GET', method_id='binaryauthorization.projects.platforms.policies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/policies', request_field='', request_type_name='BinaryauthorizationProjectsPlatformsPoliciesListRequest', response_type_name='ListPlatformPoliciesResponse', supports_download=False)

    def ReplacePlatformPolicy(self, request, global_params=None):
        """Replaces a platform policy. Returns `NOT_FOUND` if the policy doesn't exist.

      Args:
        request: (PlatformPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PlatformPolicy) The response message.
      """
        config = self.GetMethodConfig('ReplacePlatformPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    ReplacePlatformPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/platforms/{platformsId}/policies/{policiesId}', http_method='PUT', method_id='binaryauthorization.projects.platforms.policies.replacePlatformPolicy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='PlatformPolicy', response_type_name='PlatformPolicy', supports_download=False)