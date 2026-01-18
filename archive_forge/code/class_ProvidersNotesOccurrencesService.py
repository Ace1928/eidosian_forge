from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1alpha1 import containeranalysis_v1alpha1_messages as messages
class ProvidersNotesOccurrencesService(base_api.BaseApiService):
    """Service class for the providers_notes_occurrences resource."""
    _NAME = 'providers_notes_occurrences'

    def __init__(self, client):
        super(ContaineranalysisV1alpha1.ProvidersNotesOccurrencesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists `Occurrences` referencing the specified `Note`. Use this method to get all occurrences referencing your `Note` across all your customer projects.

      Args:
        request: (ContaineranalysisProvidersNotesOccurrencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNoteOccurrencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes/{notesId}/occurrences', http_method='GET', method_id='containeranalysis.providers.notes.occurrences.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+name}/occurrences', request_field='', request_type_name='ContaineranalysisProvidersNotesOccurrencesListRequest', response_type_name='ListNoteOccurrencesResponse', supports_download=False)