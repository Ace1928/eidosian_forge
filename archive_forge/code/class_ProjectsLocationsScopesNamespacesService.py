from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsScopesNamespacesService(base_api.BaseApiService):
    """Service class for the projects_locations_scopes_namespaces resource."""
    _NAME = 'projects_locations_scopes_namespaces'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsScopesNamespacesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a fleet namespace.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces', http_method='POST', method_id='gkehub.projects.locations.scopes.namespaces.create', ordered_params=['parent'], path_params=['parent'], query_params=['scopeNamespaceId'], relative_path='v1beta/{+parent}/namespaces', request_field='namespace', request_type_name='GkehubProjectsLocationsScopesNamespacesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a fleet namespace.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}', http_method='DELETE', method_id='gkehub.projects.locations.scopes.namespaces.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesNamespacesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details of a fleet namespace.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Namespace) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}', http_method='GET', method_id='gkehub.projects.locations.scopes.namespaces.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesNamespacesGetRequest', response_type_name='Namespace', supports_download=False)

    def List(self, request, global_params=None):
        """Lists fleet namespaces.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScopeNamespacesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces', http_method='GET', method_id='gkehub.projects.locations.scopes.namespaces.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/namespaces', request_field='', request_type_name='GkehubProjectsLocationsScopesNamespacesListRequest', response_type_name='ListScopeNamespacesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a fleet namespace.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}', http_method='PATCH', method_id='gkehub.projects.locations.scopes.namespaces.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='namespace', request_type_name='GkehubProjectsLocationsScopesNamespacesPatchRequest', response_type_name='Operation', supports_download=False)