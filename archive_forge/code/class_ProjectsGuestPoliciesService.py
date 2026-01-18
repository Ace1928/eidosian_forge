from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1beta import osconfig_v1beta_messages as messages
class ProjectsGuestPoliciesService(base_api.BaseApiService):
    """Service class for the projects_guestPolicies resource."""
    _NAME = 'projects_guestPolicies'

    def __init__(self, client):
        super(OsconfigV1beta.ProjectsGuestPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an OS Config guest policy.

      Args:
        request: (OsconfigProjectsGuestPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GuestPolicy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/guestPolicies', http_method='POST', method_id='osconfig.projects.guestPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['guestPolicyId'], relative_path='v1beta/{+parent}/guestPolicies', request_field='guestPolicy', request_type_name='OsconfigProjectsGuestPoliciesCreateRequest', response_type_name='GuestPolicy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an OS Config guest policy.

      Args:
        request: (OsconfigProjectsGuestPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/guestPolicies/{guestPoliciesId}', http_method='DELETE', method_id='osconfig.projects.guestPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsconfigProjectsGuestPoliciesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get an OS Config guest policy.

      Args:
        request: (OsconfigProjectsGuestPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GuestPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/guestPolicies/{guestPoliciesId}', http_method='GET', method_id='osconfig.projects.guestPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsconfigProjectsGuestPoliciesGetRequest', response_type_name='GuestPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Get a page of OS Config guest policies.

      Args:
        request: (OsconfigProjectsGuestPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGuestPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/guestPolicies', http_method='GET', method_id='osconfig.projects.guestPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/guestPolicies', request_field='', request_type_name='OsconfigProjectsGuestPoliciesListRequest', response_type_name='ListGuestPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an OS Config guest policy.

      Args:
        request: (OsconfigProjectsGuestPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GuestPolicy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/guestPolicies/{guestPoliciesId}', http_method='PATCH', method_id='osconfig.projects.guestPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='guestPolicy', request_type_name='OsconfigProjectsGuestPoliciesPatchRequest', response_type_name='GuestPolicy', supports_download=False)