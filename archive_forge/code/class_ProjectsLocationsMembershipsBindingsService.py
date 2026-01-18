from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsMembershipsBindingsService(base_api.BaseApiService):
    """Service class for the projects_locations_memberships_bindings resource."""
    _NAME = 'projects_locations_memberships_bindings'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsMembershipsBindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a MembershipBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsBindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/bindings', http_method='POST', method_id='gkehub.projects.locations.memberships.bindings.create', ordered_params=['parent'], path_params=['parent'], query_params=['membershipBindingId'], relative_path='v1beta/{+parent}/bindings', request_field='membershipBinding', request_type_name='GkehubProjectsLocationsMembershipsBindingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a MembershipBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsBindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/bindings/{bindingsId}', http_method='DELETE', method_id='gkehub.projects.locations.memberships.bindings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsMembershipsBindingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details of a MembershipBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsBindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MembershipBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/bindings/{bindingsId}', http_method='GET', method_id='gkehub.projects.locations.memberships.bindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsMembershipsBindingsGetRequest', response_type_name='MembershipBinding', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MembershipBindings.

      Args:
        request: (GkehubProjectsLocationsMembershipsBindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/bindings', http_method='GET', method_id='gkehub.projects.locations.memberships.bindings.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/bindings', request_field='', request_type_name='GkehubProjectsLocationsMembershipsBindingsListRequest', response_type_name='ListMembershipBindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a MembershipBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsBindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/bindings/{bindingsId}', http_method='PATCH', method_id='gkehub.projects.locations.memberships.bindings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='membershipBinding', request_type_name='GkehubProjectsLocationsMembershipsBindingsPatchRequest', response_type_name='Operation', supports_download=False)