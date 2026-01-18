from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v2 import speech_v2_messages as messages
class ProjectsLocationsPhraseSetsService(base_api.BaseApiService):
    """Service class for the projects_locations_phraseSets resource."""
    _NAME = 'projects_locations_phraseSets'

    def __init__(self, client):
        super(SpeechV2.ProjectsLocationsPhraseSetsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a PhraseSet.

      Args:
        request: (SpeechProjectsLocationsPhraseSetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/phraseSets', http_method='POST', method_id='speech.projects.locations.phraseSets.create', ordered_params=['parent'], path_params=['parent'], query_params=['phraseSetId', 'validateOnly'], relative_path='v2/{+parent}/phraseSets', request_field='phraseSet', request_type_name='SpeechProjectsLocationsPhraseSetsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the PhraseSet.

      Args:
        request: (SpeechProjectsLocationsPhraseSetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/phraseSets/{phraseSetsId}', http_method='DELETE', method_id='speech.projects.locations.phraseSets.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='SpeechProjectsLocationsPhraseSetsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested PhraseSet.

      Args:
        request: (SpeechProjectsLocationsPhraseSetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PhraseSet) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/phraseSets/{phraseSetsId}', http_method='GET', method_id='speech.projects.locations.phraseSets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SpeechProjectsLocationsPhraseSetsGetRequest', response_type_name='PhraseSet', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PhraseSets.

      Args:
        request: (SpeechProjectsLocationsPhraseSetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPhraseSetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/phraseSets', http_method='GET', method_id='speech.projects.locations.phraseSets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/phraseSets', request_field='', request_type_name='SpeechProjectsLocationsPhraseSetsListRequest', response_type_name='ListPhraseSetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the PhraseSet.

      Args:
        request: (SpeechProjectsLocationsPhraseSetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/phraseSets/{phraseSetsId}', http_method='PATCH', method_id='speech.projects.locations.phraseSets.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v2/{+name}', request_field='phraseSet', request_type_name='SpeechProjectsLocationsPhraseSetsPatchRequest', response_type_name='Operation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes the PhraseSet.

      Args:
        request: (UndeletePhraseSetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/phraseSets/{phraseSetsId}:undelete', http_method='POST', method_id='speech.projects.locations.phraseSets.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:undelete', request_field='<request>', request_type_name='UndeletePhraseSetRequest', response_type_name='Operation', supports_download=False)