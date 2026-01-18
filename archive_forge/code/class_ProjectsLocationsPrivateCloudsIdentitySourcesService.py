from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsIdentitySourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_identitySources resource."""
    _NAME = 'projects_locations_privateClouds_identitySources'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsIdentitySourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new identity source in a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/identitySources', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.identitySources.create', ordered_params=['parent'], path_params=['parent'], query_params=['identitySourceId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/identitySources', request_field='identitySource', request_type_name='VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `IdentitySource` resource.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/identitySources/{identitySourcesId}', http_method='DELETE', method_id='vmwareengine.projects.locations.privateClouds.identitySources.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the `IdentitySource` resource by its resource name.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IdentitySource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/identitySources/{identitySourcesId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.identitySources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesGetRequest', response_type_name='IdentitySource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists identity sources in the private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListIdentitySourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/identitySources', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.identitySources.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/identitySources', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesListRequest', response_type_name='ListIdentitySourcesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Modifies an `IdentitySource` resource. Only the following fields can be updated: `base_users_dn`, `base_groups_dn`, `domain_user`, `domain_password` and `ssl_certificates`. Only fields specified in `update_mask` are applied. When updating identity source with LDAPS protocol, update mask must include `ssl_certificates`. When updating identity source with LDAP protocol, update mask must not include `ssl_certificates`. When updating `domain_user`, `domain_password` must be updated as well, and the other way around.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/identitySources/{identitySourcesId}', http_method='PATCH', method_id='vmwareengine.projects.locations.privateClouds.identitySources.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='identitySource', request_type_name='VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesPatchRequest', response_type_name='Operation', supports_download=False)