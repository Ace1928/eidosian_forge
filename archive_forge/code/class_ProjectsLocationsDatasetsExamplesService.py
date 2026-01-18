from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3 import translate_v3_messages as messages
class ProjectsLocationsDatasetsExamplesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_examples resource."""
    _NAME = 'projects_locations_datasets_examples'

    def __init__(self, client):
        super(TranslateV3.ProjectsLocationsDatasetsExamplesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists sentence pairs in the dataset.

      Args:
        request: (TranslateProjectsLocationsDatasetsExamplesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExamplesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/examples', http_method='GET', method_id='translate.projects.locations.datasets.examples.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3/{+parent}/examples', request_field='', request_type_name='TranslateProjectsLocationsDatasetsExamplesListRequest', response_type_name='ListExamplesResponse', supports_download=False)