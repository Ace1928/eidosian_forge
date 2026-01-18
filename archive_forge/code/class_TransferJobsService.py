from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storagetransfer.v1 import storagetransfer_v1_messages as messages
class TransferJobsService(base_api.BaseApiService):
    """Service class for the transferJobs resource."""
    _NAME = 'transferJobs'

    def __init__(self, client):
        super(StoragetransferV1.TransferJobsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a transfer job that runs periodically.

      Args:
        request: (TransferJob) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TransferJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='storagetransfer.transferJobs.create', ordered_params=[], path_params=[], query_params=[], relative_path='v1/transferJobs', request_field='<request>', request_type_name='TransferJob', response_type_name='TransferJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a transfer job. Deleting a transfer job sets its status to DELETED.

      Args:
        request: (StoragetransferTransferJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferJobs/{transferJobsId}', http_method='DELETE', method_id='storagetransfer.transferJobs.delete', ordered_params=['jobName', 'projectId'], path_params=['jobName'], query_params=['projectId'], relative_path='v1/{+jobName}', request_field='', request_type_name='StoragetransferTransferJobsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a transfer job.

      Args:
        request: (StoragetransferTransferJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TransferJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferJobs/{transferJobsId}', http_method='GET', method_id='storagetransfer.transferJobs.get', ordered_params=['jobName', 'projectId'], path_params=['jobName'], query_params=['projectId'], relative_path='v1/{+jobName}', request_field='', request_type_name='StoragetransferTransferJobsGetRequest', response_type_name='TransferJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists transfer jobs.

      Args:
        request: (StoragetransferTransferJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTransferJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storagetransfer.transferJobs.list', ordered_params=['filter'], path_params=[], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/transferJobs', request_field='', request_type_name='StoragetransferTransferJobsListRequest', response_type_name='ListTransferJobsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a transfer job. Updating a job's transfer spec does not affect transfer operations that are running already. **Note:** The job's status field can be modified using this RPC (for example, to set a job's status to DELETED, DISABLED, or ENABLED).

      Args:
        request: (StoragetransferTransferJobsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TransferJob) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferJobs/{transferJobsId}', http_method='PATCH', method_id='storagetransfer.transferJobs.patch', ordered_params=['jobName'], path_params=['jobName'], query_params=[], relative_path='v1/{+jobName}', request_field='updateTransferJobRequest', request_type_name='StoragetransferTransferJobsPatchRequest', response_type_name='TransferJob', supports_download=False)

    def Run(self, request, global_params=None):
        """Starts a new operation for the specified transfer job. A `TransferJob` has a maximum of one active `TransferOperation`. If this method is called while a `TransferOperation` is active, an error is returned.

      Args:
        request: (StoragetransferTransferJobsRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferJobs/{transferJobsId}:run', http_method='POST', method_id='storagetransfer.transferJobs.run', ordered_params=['jobName'], path_params=['jobName'], query_params=[], relative_path='v1/{+jobName}:run', request_field='runTransferJobRequest', request_type_name='StoragetransferTransferJobsRunRequest', response_type_name='Operation', supports_download=False)