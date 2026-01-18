from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsIndexesService(base_api.BaseApiService):
    """Service class for the projects_locations_indexes resource."""
    _NAME = 'projects_locations_indexes'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsIndexesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an Index.

      Args:
        request: (AiplatformProjectsLocationsIndexesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexes', http_method='POST', method_id='aiplatform.projects.locations.indexes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/indexes', request_field='googleCloudAiplatformV1Index', request_type_name='AiplatformProjectsLocationsIndexesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Index. An Index can only be deleted when all its DeployedIndexes had been undeployed.

      Args:
        request: (AiplatformProjectsLocationsIndexesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexes/{indexesId}', http_method='DELETE', method_id='aiplatform.projects.locations.indexes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsIndexesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an Index.

      Args:
        request: (AiplatformProjectsLocationsIndexesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Index) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexes/{indexesId}', http_method='GET', method_id='aiplatform.projects.locations.indexes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsIndexesGetRequest', response_type_name='GoogleCloudAiplatformV1Index', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Indexes in a Location.

      Args:
        request: (AiplatformProjectsLocationsIndexesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListIndexesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexes', http_method='GET', method_id='aiplatform.projects.locations.indexes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/indexes', request_field='', request_type_name='AiplatformProjectsLocationsIndexesListRequest', response_type_name='GoogleCloudAiplatformV1ListIndexesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an Index.

      Args:
        request: (AiplatformProjectsLocationsIndexesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexes/{indexesId}', http_method='PATCH', method_id='aiplatform.projects.locations.indexes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Index', request_type_name='AiplatformProjectsLocationsIndexesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def RemoveDatapoints(self, request, global_params=None):
        """Remove Datapoints from an Index.

      Args:
        request: (AiplatformProjectsLocationsIndexesRemoveDatapointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1RemoveDatapointsResponse) The response message.
      """
        config = self.GetMethodConfig('RemoveDatapoints')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveDatapoints.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexes/{indexesId}:removeDatapoints', http_method='POST', method_id='aiplatform.projects.locations.indexes.removeDatapoints', ordered_params=['index'], path_params=['index'], query_params=[], relative_path='v1/{+index}:removeDatapoints', request_field='googleCloudAiplatformV1RemoveDatapointsRequest', request_type_name='AiplatformProjectsLocationsIndexesRemoveDatapointsRequest', response_type_name='GoogleCloudAiplatformV1RemoveDatapointsResponse', supports_download=False)

    def UpsertDatapoints(self, request, global_params=None):
        """Add/update Datapoints into an Index.

      Args:
        request: (AiplatformProjectsLocationsIndexesUpsertDatapointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1UpsertDatapointsResponse) The response message.
      """
        config = self.GetMethodConfig('UpsertDatapoints')
        return self._RunMethod(config, request, global_params=global_params)
    UpsertDatapoints.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/indexes/{indexesId}:upsertDatapoints', http_method='POST', method_id='aiplatform.projects.locations.indexes.upsertDatapoints', ordered_params=['index'], path_params=['index'], query_params=[], relative_path='v1/{+index}:upsertDatapoints', request_field='googleCloudAiplatformV1UpsertDatapointsRequest', request_type_name='AiplatformProjectsLocationsIndexesUpsertDatapointsRequest', response_type_name='GoogleCloudAiplatformV1UpsertDatapointsResponse', supports_download=False)