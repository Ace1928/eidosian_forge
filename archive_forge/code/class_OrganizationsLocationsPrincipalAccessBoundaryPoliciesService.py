from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
class OrganizationsLocationsPrincipalAccessBoundaryPoliciesService(base_api.BaseApiService):
    """Service class for the organizations_locations_principalAccessBoundaryPolicies resource."""
    _NAME = 'organizations_locations_principalAccessBoundaryPolicies'

    def __init__(self, client):
        super(IamV3beta.OrganizationsLocationsPrincipalAccessBoundaryPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a principal access boundary policy, and returns a long running operation.

      Args:
        request: (IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/organizations/{organizationsId}/locations/{locationsId}/principalAccessBoundaryPolicies', http_method='POST', method_id='iam.organizations.locations.principalAccessBoundaryPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['principalAccessBoundaryPolicyId', 'validateOnly'], relative_path='v3beta/{+parent}/principalAccessBoundaryPolicies', request_field='googleIamV3betaPrincipalAccessBoundaryPolicy', request_type_name='IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a principal access boundary policy.

      Args:
        request: (IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/organizations/{organizationsId}/locations/{locationsId}/principalAccessBoundaryPolicies/{principalAccessBoundaryPoliciesId}', http_method='DELETE', method_id='iam.organizations.locations.principalAccessBoundaryPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force', 'validateOnly'], relative_path='v3beta/{+name}', request_field='', request_type_name='IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a principal access boundary policy.

      Args:
        request: (IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaPrincipalAccessBoundaryPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/organizations/{organizationsId}/locations/{locationsId}/principalAccessBoundaryPolicies/{principalAccessBoundaryPoliciesId}', http_method='GET', method_id='iam.organizations.locations.principalAccessBoundaryPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3beta/{+name}', request_field='', request_type_name='IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesGetRequest', response_type_name='GoogleIamV3betaPrincipalAccessBoundaryPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists principal access boundary policies.

      Args:
        request: (IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaListPrincipalAccessBoundaryPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/organizations/{organizationsId}/locations/{locationsId}/principalAccessBoundaryPolicies', http_method='GET', method_id='iam.organizations.locations.principalAccessBoundaryPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v3beta/{+parent}/principalAccessBoundaryPolicies', request_field='', request_type_name='IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesListRequest', response_type_name='GoogleIamV3betaListPrincipalAccessBoundaryPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a principal access boundary policy.

      Args:
        request: (IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/organizations/{organizationsId}/locations/{locationsId}/principalAccessBoundaryPolicies/{principalAccessBoundaryPoliciesId}', http_method='PATCH', method_id='iam.organizations.locations.principalAccessBoundaryPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v3beta/{+name}', request_field='googleIamV3betaPrincipalAccessBoundaryPolicy', request_type_name='IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SearchPolicyBindings(self, request, global_params=None):
        """Returns all policy bindings that bind a specific policy if a user has searchPolicyBindings permission on that policy.

      Args:
        request: (IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesSearchPolicyBindingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaSearchPrincipalAccessBoundaryPolicyBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchPolicyBindings')
        return self._RunMethod(config, request, global_params=global_params)
    SearchPolicyBindings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta/organizations/{organizationsId}/locations/{locationsId}/principalAccessBoundaryPolicies/{principalAccessBoundaryPoliciesId}:searchPolicyBindings', http_method='GET', method_id='iam.organizations.locations.principalAccessBoundaryPolicies.searchPolicyBindings', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v3beta/{+name}:searchPolicyBindings', request_field='', request_type_name='IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesSearchPolicyBindingsRequest', response_type_name='GoogleIamV3betaSearchPrincipalAccessBoundaryPolicyBindingsResponse', supports_download=False)