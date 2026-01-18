from __future__ import absolute_import
from apitools.base.py import base_api
from samples.bigquery_sample.bigquery_v2 import bigquery_v2_messages as messages
class DatasetsService(base_api.BaseApiService):
    """Service class for the datasets resource."""
    _NAME = u'datasets'

    def __init__(self, client):
        super(BigqueryV2.DatasetsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the dataset specified by the datasetId value. Before you can delete a dataset, you must delete all its tables, either manually or by specifying deleteContents. Immediately after deletion, you can create another dataset with the same name.

      Args:
        request: (BigqueryDatasetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BigqueryDatasetsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'bigquery.datasets.delete', ordered_params=[u'projectId', u'datasetId'], path_params=[u'datasetId', u'projectId'], query_params=[u'deleteContents'], relative_path=u'projects/{projectId}/datasets/{datasetId}', request_field='', request_type_name=u'BigqueryDatasetsDeleteRequest', response_type_name=u'BigqueryDatasetsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the dataset specified by datasetID.

      Args:
        request: (BigqueryDatasetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dataset) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.datasets.get', ordered_params=[u'projectId', u'datasetId'], path_params=[u'datasetId', u'projectId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}', request_field='', request_type_name=u'BigqueryDatasetsGetRequest', response_type_name=u'Dataset', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new empty dataset.

      Args:
        request: (BigqueryDatasetsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dataset) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'bigquery.datasets.insert', ordered_params=[u'projectId'], path_params=[u'projectId'], query_params=[], relative_path=u'projects/{projectId}/datasets', request_field=u'dataset', request_type_name=u'BigqueryDatasetsInsertRequest', response_type_name=u'Dataset', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all datasets in the specified project to which you have been granted the READER dataset role.

      Args:
        request: (BigqueryDatasetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DatasetList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.datasets.list', ordered_params=[u'projectId'], path_params=[u'projectId'], query_params=[u'all', u'filter', u'maxResults', u'pageToken'], relative_path=u'projects/{projectId}/datasets', request_field='', request_type_name=u'BigqueryDatasetsListRequest', response_type_name=u'DatasetList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates information in an existing dataset. The update method replaces the entire dataset resource, whereas the patch method only replaces fields that are provided in the submitted dataset resource. This method supports patch semantics.

      Args:
        request: (BigqueryDatasetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dataset) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'bigquery.datasets.patch', ordered_params=[u'projectId', u'datasetId'], path_params=[u'datasetId', u'projectId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}', request_field=u'dataset', request_type_name=u'BigqueryDatasetsPatchRequest', response_type_name=u'Dataset', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates information in an existing dataset. The update method replaces the entire dataset resource, whereas the patch method only replaces fields that are provided in the submitted dataset resource.

      Args:
        request: (BigqueryDatasetsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dataset) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'bigquery.datasets.update', ordered_params=[u'projectId', u'datasetId'], path_params=[u'datasetId', u'projectId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}', request_field=u'dataset', request_type_name=u'BigqueryDatasetsUpdateRequest', response_type_name=u'Dataset', supports_download=False)