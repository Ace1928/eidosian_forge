from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3beta1 import translate_v3beta1_messages as messages
class ProjectsLocationsGlossariesService(base_api.BaseApiService):
    """Service class for the projects_locations_glossaries resource."""
    _NAME = 'projects_locations_glossaries'

    def __init__(self, client):
        super(TranslateV3beta1.ProjectsLocationsGlossariesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a glossary and returns the long-running operation. Returns NOT_FOUND, if the project doesn't exist.

      Args:
        request: (TranslateProjectsLocationsGlossariesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta1/projects/{projectsId}/locations/{locationsId}/glossaries', http_method='POST', method_id='translate.projects.locations.glossaries.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v3beta1/{+parent}/glossaries', request_field='glossary', request_type_name='TranslateProjectsLocationsGlossariesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a glossary, or cancels glossary construction if the glossary isn't created yet. Returns NOT_FOUND, if the glossary doesn't exist.

      Args:
        request: (TranslateProjectsLocationsGlossariesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta1/projects/{projectsId}/locations/{locationsId}/glossaries/{glossariesId}', http_method='DELETE', method_id='translate.projects.locations.glossaries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3beta1/{+name}', request_field='', request_type_name='TranslateProjectsLocationsGlossariesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a glossary. Returns NOT_FOUND, if the glossary doesn't exist.

      Args:
        request: (TranslateProjectsLocationsGlossariesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Glossary) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta1/projects/{projectsId}/locations/{locationsId}/glossaries/{glossariesId}', http_method='GET', method_id='translate.projects.locations.glossaries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3beta1/{+name}', request_field='', request_type_name='TranslateProjectsLocationsGlossariesGetRequest', response_type_name='Glossary', supports_download=False)

    def List(self, request, global_params=None):
        """Lists glossaries in a project. Returns NOT_FOUND, if the project doesn't exist.

      Args:
        request: (TranslateProjectsLocationsGlossariesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGlossariesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3beta1/projects/{projectsId}/locations/{locationsId}/glossaries', http_method='GET', method_id='translate.projects.locations.glossaries.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3beta1/{+parent}/glossaries', request_field='', request_type_name='TranslateProjectsLocationsGlossariesListRequest', response_type_name='ListGlossariesResponse', supports_download=False)