from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composerflex.v1alpha1 import composerflex_v1alpha1_messages as messages
class ProjectsLocationsContextsService(base_api.BaseApiService):
    """Service class for the projects_locations_contexts resource."""
    _NAME = 'projects_locations_contexts'

    def __init__(self, client):
        super(ComposerflexV1alpha1.ProjectsLocationsContextsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new context.

      Args:
        request: (ComposerflexProjectsLocationsContextsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/contexts', http_method='POST', method_id='composerflex.projects.locations.contexts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/contexts', request_field='context', request_type_name='ComposerflexProjectsLocationsContextsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a context. A context cannot be deleted if any workflows are bound to it.

      Args:
        request: (ComposerflexProjectsLocationsContextsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/contexts/{contextsId}', http_method='DELETE', method_id='composerflex.projects.locations.contexts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='ComposerflexProjectsLocationsContextsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a context.

      Args:
        request: (ComposerflexProjectsLocationsContextsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Context) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/contexts/{contextsId}', http_method='GET', method_id='composerflex.projects.locations.contexts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='ComposerflexProjectsLocationsContextsGetRequest', response_type_name='Context', supports_download=False)

    def List(self, request, global_params=None):
        """Lists contexts within a project and location.

      Args:
        request: (ComposerflexProjectsLocationsContextsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListContextsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/contexts', http_method='GET', method_id='composerflex.projects.locations.contexts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/contexts', request_field='', request_type_name='ComposerflexProjectsLocationsContextsListRequest', response_type_name='ListContextsResponse', supports_download=False)