from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsTensorboardsService(base_api.BaseApiService):
    """Service class for the projects_locations_tensorboards resource."""
    _NAME = 'projects_locations_tensorboards'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsTensorboardsService, self).__init__(client)
        self._upload_configs = {}

    def BatchRead(self, request, global_params=None):
        """Reads multiple TensorboardTimeSeries' data. The data point number limit is 1000 for scalars, 100 for tensors and blob references. If the number of data points stored is less than the limit, all data is returned. Otherwise, the number limit of data points is randomly selected from this time series and returned.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsBatchReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1BatchReadTensorboardTimeSeriesDataResponse) The response message.
      """
        config = self.GetMethodConfig('BatchRead')
        return self._RunMethod(config, request, global_params=global_params)
    BatchRead.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}:batchRead', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.batchRead', ordered_params=['tensorboard'], path_params=['tensorboard'], query_params=['timeSeries'], relative_path='v1/{+tensorboard}:batchRead', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsBatchReadRequest', response_type_name='GoogleCloudAiplatformV1BatchReadTensorboardTimeSeriesDataResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a Tensorboard.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards', http_method='POST', method_id='aiplatform.projects.locations.tensorboards.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/tensorboards', request_field='googleCloudAiplatformV1Tensorboard', request_type_name='AiplatformProjectsLocationsTensorboardsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Tensorboard.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}', http_method='DELETE', method_id='aiplatform.projects.locations.tensorboards.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Tensorboard.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Tensorboard) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsGetRequest', response_type_name='GoogleCloudAiplatformV1Tensorboard', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Tensorboards in a Location.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListTensorboardsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/tensorboards', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsListRequest', response_type_name='GoogleCloudAiplatformV1ListTensorboardsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Tensorboard.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}', http_method='PATCH', method_id='aiplatform.projects.locations.tensorboards.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Tensorboard', request_type_name='AiplatformProjectsLocationsTensorboardsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ReadSize(self, request, global_params=None):
        """Returns the storage size for a given TensorBoard instance.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsReadSizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadTensorboardSizeResponse) The response message.
      """
        config = self.GetMethodConfig('ReadSize')
        return self._RunMethod(config, request, global_params=global_params)
    ReadSize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}:readSize', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.readSize', ordered_params=['tensorboard'], path_params=['tensorboard'], query_params=[], relative_path='v1/{+tensorboard}:readSize', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsReadSizeRequest', response_type_name='GoogleCloudAiplatformV1ReadTensorboardSizeResponse', supports_download=False)

    def ReadUsage(self, request, global_params=None):
        """Returns a list of monthly active users for a given TensorBoard instance.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsReadUsageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadTensorboardUsageResponse) The response message.
      """
        config = self.GetMethodConfig('ReadUsage')
        return self._RunMethod(config, request, global_params=global_params)
    ReadUsage.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}:readUsage', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.readUsage', ordered_params=['tensorboard'], path_params=['tensorboard'], query_params=[], relative_path='v1/{+tensorboard}:readUsage', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsReadUsageRequest', response_type_name='GoogleCloudAiplatformV1ReadTensorboardUsageResponse', supports_download=False)