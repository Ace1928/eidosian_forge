from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDatasetsDataItemsAnnotationsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dataItems_annotations resource."""
    _NAME = 'projects_locations_datasets_dataItems_annotations'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDatasetsDataItemsAnnotationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists Annotations belongs to a dataitem.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDataItemsAnnotationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListAnnotationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dataItems/{dataItemsId}/annotations', http_method='GET', method_id='aiplatform.projects.locations.datasets.dataItems.annotations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/annotations', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsDataItemsAnnotationsListRequest', response_type_name='GoogleCloudAiplatformV1ListAnnotationsResponse', supports_download=False)