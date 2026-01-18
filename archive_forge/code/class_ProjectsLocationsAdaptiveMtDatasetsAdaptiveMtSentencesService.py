from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3 import translate_v3_messages as messages
class ProjectsLocationsAdaptiveMtDatasetsAdaptiveMtSentencesService(base_api.BaseApiService):
    """Service class for the projects_locations_adaptiveMtDatasets_adaptiveMtSentences resource."""
    _NAME = 'projects_locations_adaptiveMtDatasets_adaptiveMtSentences'

    def __init__(self, client):
        super(TranslateV3.ProjectsLocationsAdaptiveMtDatasetsAdaptiveMtSentencesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all AdaptiveMtSentences under a given file/dataset.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtSentencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAdaptiveMtSentencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets/{adaptiveMtDatasetsId}/adaptiveMtSentences', http_method='GET', method_id='translate.projects.locations.adaptiveMtDatasets.adaptiveMtSentences.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v3/{+parent}/adaptiveMtSentences', request_field='', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtSentencesListRequest', response_type_name='ListAdaptiveMtSentencesResponse', supports_download=False)