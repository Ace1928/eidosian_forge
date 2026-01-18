from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3 import translate_v3_messages as messages
class ProjectsLocationsGlossariesGlossaryEntriesService(base_api.BaseApiService):
    """Service class for the projects_locations_glossaries_glossaryEntries resource."""
    _NAME = 'projects_locations_glossaries_glossaryEntries'

    def __init__(self, client):
        super(TranslateV3.ProjectsLocationsGlossariesGlossaryEntriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a glossary entry.

      Args:
        request: (TranslateProjectsLocationsGlossariesGlossaryEntriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GlossaryEntry) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/glossaries/{glossariesId}/glossaryEntries', http_method='POST', method_id='translate.projects.locations.glossaries.glossaryEntries.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v3/{+parent}/glossaryEntries', request_field='glossaryEntry', request_type_name='TranslateProjectsLocationsGlossariesGlossaryEntriesCreateRequest', response_type_name='GlossaryEntry', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single entry from the glossary.

      Args:
        request: (TranslateProjectsLocationsGlossariesGlossaryEntriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/glossaries/{glossariesId}/glossaryEntries/{glossaryEntriesId}', http_method='DELETE', method_id='translate.projects.locations.glossaries.glossaryEntries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='TranslateProjectsLocationsGlossariesGlossaryEntriesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single glossary entry by the given id.

      Args:
        request: (TranslateProjectsLocationsGlossariesGlossaryEntriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GlossaryEntry) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/glossaries/{glossariesId}/glossaryEntries/{glossaryEntriesId}', http_method='GET', method_id='translate.projects.locations.glossaries.glossaryEntries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='TranslateProjectsLocationsGlossariesGlossaryEntriesGetRequest', response_type_name='GlossaryEntry', supports_download=False)

    def List(self, request, global_params=None):
        """List the entries for the glossary.

      Args:
        request: (TranslateProjectsLocationsGlossariesGlossaryEntriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGlossaryEntriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/glossaries/{glossariesId}/glossaryEntries', http_method='GET', method_id='translate.projects.locations.glossaries.glossaryEntries.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v3/{+parent}/glossaryEntries', request_field='', request_type_name='TranslateProjectsLocationsGlossariesGlossaryEntriesListRequest', response_type_name='ListGlossaryEntriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a glossary entry.

      Args:
        request: (GlossaryEntry) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GlossaryEntry) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/glossaries/{glossariesId}/glossaryEntries/{glossaryEntriesId}', http_method='PATCH', method_id='translate.projects.locations.glossaries.glossaryEntries.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='<request>', request_type_name='GlossaryEntry', response_type_name='GlossaryEntry', supports_download=False)