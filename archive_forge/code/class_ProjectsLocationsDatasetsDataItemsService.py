from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDatasetsDataItemsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dataItems resource."""
    _NAME = 'projects_locations_datasets_dataItems'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDatasetsDataItemsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists DataItems in a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDataItemsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListDataItemsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dataItems', http_method='GET', method_id='aiplatform.projects.locations.datasets.dataItems.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/dataItems', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsDataItemsListRequest', response_type_name='GoogleCloudAiplatformV1ListDataItemsResponse', supports_download=False)