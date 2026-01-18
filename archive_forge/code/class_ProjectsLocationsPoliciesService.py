from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
class ProjectsLocationsPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_policies resource."""
    _NAME = 'projects_locations_policies'

    def __init__(self, client):
        super(IamV3beta.ProjectsLocationsPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a policy, and returns a long running operation.

      Args:
        request: (IamProjectsLocationsPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/projects/{projectsId}/locations/{locationsId}/policies', http_method='POST', method_id='iam.projects.locations.policies.create', ordered_params=['parent'], path_params=['parent'], query_params=['policyId', 'validateOnly'], relative_path='v3beta/{+parent}/policies', request_field='googleIamV3betaV3Policy', request_type_name='IamProjectsLocationsPoliciesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a policy. Policies with references to policy bindings can't be deleted unless `force` field is set to `true`, or these policy bindings are deleted.

      Args:
        request: (IamProjectsLocationsPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/projects/{projectsId}/locations/{locationsId}/policies/{policiesId}', http_method='DELETE', method_id='iam.projects.locations.policies.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force', 'validateOnly'], relative_path='v3beta/{+name}', request_field='', request_type_name='IamProjectsLocationsPoliciesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a policy.

      Args:
        request: (IamProjectsLocationsPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaV3Policy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/projects/{projectsId}/locations/{locationsId}/policies/{policiesId}', http_method='GET', method_id='iam.projects.locations.policies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3beta/{+name}', request_field='', request_type_name='IamProjectsLocationsPoliciesGetRequest', response_type_name='GoogleIamV3betaV3Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists policies.

      Args:
        request: (IamProjectsLocationsPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaListPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/projects/{projectsId}/locations/{locationsId}/policies', http_method='GET', method_id='iam.projects.locations.policies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v3beta/{+parent}/policies', request_field='', request_type_name='IamProjectsLocationsPoliciesListRequest', response_type_name='GoogleIamV3betaListPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a policy.

      Args:
        request: (IamProjectsLocationsPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/projects/{projectsId}/locations/{locationsId}/policies/{policiesId}', http_method='PATCH', method_id='iam.projects.locations.policies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v3beta/{+name}', request_field='googleIamV3betaV3Policy', request_type_name='IamProjectsLocationsPoliciesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)