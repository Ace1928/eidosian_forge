from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsScopesNamespacesResourcequotasService(base_api.BaseApiService):
    """Service class for the projects_locations_scopes_namespaces_resourcequotas resource."""
    _NAME = 'projects_locations_scopes_namespaces_resourcequotas'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsScopesNamespacesResourcequotasService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a resource quota.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesResourcequotasCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}/resourcequotas', http_method='POST', method_id='gkehub.projects.locations.scopes.namespaces.resourcequotas.create', ordered_params=['parent'], path_params=['parent'], query_params=['resourceQuotaId'], relative_path='v1beta/{+parent}/resourcequotas', request_field='resourceQuota', request_type_name='GkehubProjectsLocationsScopesNamespacesResourcequotasCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a resource quota.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesResourcequotasDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}/resourcequotas/{resourcequotasId}', http_method='DELETE', method_id='gkehub.projects.locations.scopes.namespaces.resourcequotas.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesNamespacesResourcequotasDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details of a resource quota.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesResourcequotasGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceQuota) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}/resourcequotas/{resourcequotasId}', http_method='GET', method_id='gkehub.projects.locations.scopes.namespaces.resourcequotas.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesNamespacesResourcequotasGetRequest', response_type_name='ResourceQuota', supports_download=False)

    def List(self, request, global_params=None):
        """Lists resource quotas.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesResourcequotasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResourceQuotasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}/resourcequotas', http_method='GET', method_id='gkehub.projects.locations.scopes.namespaces.resourcequotas.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/resourcequotas', request_field='', request_type_name='GkehubProjectsLocationsScopesNamespacesResourcequotasListRequest', response_type_name='ListResourceQuotasResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a resource quota.

      Args:
        request: (GkehubProjectsLocationsScopesNamespacesResourcequotasPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/namespaces/{namespacesId}/resourcequotas/{resourcequotasId}', http_method='PATCH', method_id='gkehub.projects.locations.scopes.namespaces.resourcequotas.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='resourceQuota', request_type_name='GkehubProjectsLocationsScopesNamespacesResourcequotasPatchRequest', response_type_name='Operation', supports_download=False)