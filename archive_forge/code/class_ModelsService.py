from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquery.v2 import bigquery_v2_messages as messages
class ModelsService(base_api.BaseApiService):
    """Service class for the models resource."""
    _NAME = 'models'

    def __init__(self, client):
        super(BigqueryV2.ModelsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the model specified by modelId from the dataset.

      Args:
        request: (BigqueryModelsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BigqueryModelsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models/{modelsId}', http_method='DELETE', method_id='bigquery.models.delete', ordered_params=['projectId', 'datasetId', 'modelId'], path_params=['datasetId', 'modelId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/models/{+modelId}', request_field='', request_type_name='BigqueryModelsDeleteRequest', response_type_name='BigqueryModelsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified model resource by model ID.

      Args:
        request: (BigqueryModelsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Model) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models/{modelsId}', http_method='GET', method_id='bigquery.models.get', ordered_params=['projectId', 'datasetId', 'modelId'], path_params=['datasetId', 'modelId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/models/{+modelId}', request_field='', request_type_name='BigqueryModelsGetRequest', response_type_name='Model', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all models in the specified dataset. Requires the READER dataset role. After retrieving the list of models, you can get information about a particular model by calling the models.get method.

      Args:
        request: (BigqueryModelsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListModelsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models', http_method='GET', method_id='bigquery.models.list', ordered_params=['projectId', 'datasetId'], path_params=['datasetId', 'projectId'], query_params=['maxResults', 'pageToken'], relative_path='projects/{+projectId}/datasets/{+datasetId}/models', request_field='', request_type_name='BigqueryModelsListRequest', response_type_name='ListModelsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patch specific fields in the specified model.

      Args:
        request: (BigqueryModelsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Model) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='projects/{projectsId}/datasets/{datasetsId}/models/{modelsId}', http_method='PATCH', method_id='bigquery.models.patch', ordered_params=['projectId', 'datasetId', 'modelId'], path_params=['datasetId', 'modelId', 'projectId'], query_params=[], relative_path='projects/{+projectId}/datasets/{+datasetId}/models/{+modelId}', request_field='model', request_type_name='BigqueryModelsPatchRequest', response_type_name='Model', supports_download=False)