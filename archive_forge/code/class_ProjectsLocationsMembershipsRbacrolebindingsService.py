from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsMembershipsRbacrolebindingsService(base_api.BaseApiService):
    """Service class for the projects_locations_memberships_rbacrolebindings resource."""
    _NAME = 'projects_locations_memberships_rbacrolebindings'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsMembershipsRbacrolebindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Membership RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsRbacrolebindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/rbacrolebindings', http_method='POST', method_id='gkehub.projects.locations.memberships.rbacrolebindings.create', ordered_params=['parent'], path_params=['parent'], query_params=['rbacrolebindingId'], relative_path='v1beta/{+parent}/rbacrolebindings', request_field='rBACRoleBinding', request_type_name='GkehubProjectsLocationsMembershipsRbacrolebindingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Membership RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsRbacrolebindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/rbacrolebindings/{rbacrolebindingsId}', http_method='DELETE', method_id='gkehub.projects.locations.memberships.rbacrolebindings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsMembershipsRbacrolebindingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def GenerateMembershipRBACRoleBindingYAML(self, request, global_params=None):
        """Generates a YAML of the RBAC policies for the specified RoleBinding and its associated impersonation resources.

      Args:
        request: (GkehubProjectsLocationsMembershipsRbacrolebindingsGenerateMembershipRBACRoleBindingYAMLRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateMembershipRBACRoleBindingYAMLResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateMembershipRBACRoleBindingYAML')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateMembershipRBACRoleBindingYAML.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/rbacrolebindings:generateMembershipRBACRoleBindingYAML', http_method='POST', method_id='gkehub.projects.locations.memberships.rbacrolebindings.generateMembershipRBACRoleBindingYAML', ordered_params=['parent'], path_params=['parent'], query_params=['rbacrolebindingId'], relative_path='v1beta/{+parent}/rbacrolebindings:generateMembershipRBACRoleBindingYAML', request_field='rBACRoleBinding', request_type_name='GkehubProjectsLocationsMembershipsRbacrolebindingsGenerateMembershipRBACRoleBindingYAMLRequest', response_type_name='GenerateMembershipRBACRoleBindingYAMLResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details of a Membership RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsRbacrolebindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RBACRoleBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/rbacrolebindings/{rbacrolebindingsId}', http_method='GET', method_id='gkehub.projects.locations.memberships.rbacrolebindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsMembershipsRbacrolebindingsGetRequest', response_type_name='RBACRoleBinding', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all Membership RBACRoleBindings.

      Args:
        request: (GkehubProjectsLocationsMembershipsRbacrolebindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipRBACRoleBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/rbacrolebindings', http_method='GET', method_id='gkehub.projects.locations.memberships.rbacrolebindings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/rbacrolebindings', request_field='', request_type_name='GkehubProjectsLocationsMembershipsRbacrolebindingsListRequest', response_type_name='ListMembershipRBACRoleBindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Membership RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsMembershipsRbacrolebindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/memberships/{membershipsId}/rbacrolebindings/{rbacrolebindingsId}', http_method='PATCH', method_id='gkehub.projects.locations.memberships.rbacrolebindings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='rBACRoleBinding', request_type_name='GkehubProjectsLocationsMembershipsRbacrolebindingsPatchRequest', response_type_name='Operation', supports_download=False)