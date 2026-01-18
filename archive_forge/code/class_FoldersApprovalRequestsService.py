from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accessapproval.v1 import accessapproval_v1_messages as messages
class FoldersApprovalRequestsService(base_api.BaseApiService):
    """Service class for the folders_approvalRequests resource."""
    _NAME = 'folders_approvalRequests'

    def __init__(self, client):
        super(AccessapprovalV1.FoldersApprovalRequestsService, self).__init__(client)
        self._upload_configs = {}

    def Approve(self, request, global_params=None):
        """Approves a request and returns the updated ApprovalRequest. Returns NOT_FOUND if the request does not exist. Returns FAILED_PRECONDITION if the request exists but is not in a pending state.

      Args:
        request: (AccessapprovalFoldersApprovalRequestsApproveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApprovalRequest) The response message.
      """
        config = self.GetMethodConfig('Approve')
        return self._RunMethod(config, request, global_params=global_params)
    Approve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/approvalRequests/{approvalRequestsId}:approve', http_method='POST', method_id='accessapproval.folders.approvalRequests.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:approve', request_field='approveApprovalRequestMessage', request_type_name='AccessapprovalFoldersApprovalRequestsApproveRequest', response_type_name='ApprovalRequest', supports_download=False)

    def Dismiss(self, request, global_params=None):
        """Dismisses a request. Returns the updated ApprovalRequest. NOTE: This does not deny access to the resource if another request has been made and approved. It is equivalent in effect to ignoring the request altogether. Returns NOT_FOUND if the request does not exist. Returns FAILED_PRECONDITION if the request exists but is not in a pending state.

      Args:
        request: (AccessapprovalFoldersApprovalRequestsDismissRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApprovalRequest) The response message.
      """
        config = self.GetMethodConfig('Dismiss')
        return self._RunMethod(config, request, global_params=global_params)
    Dismiss.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/approvalRequests/{approvalRequestsId}:dismiss', http_method='POST', method_id='accessapproval.folders.approvalRequests.dismiss', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:dismiss', request_field='dismissApprovalRequestMessage', request_type_name='AccessapprovalFoldersApprovalRequestsDismissRequest', response_type_name='ApprovalRequest', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an approval request. Returns NOT_FOUND if the request does not exist.

      Args:
        request: (AccessapprovalFoldersApprovalRequestsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApprovalRequest) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/approvalRequests/{approvalRequestsId}', http_method='GET', method_id='accessapproval.folders.approvalRequests.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AccessapprovalFoldersApprovalRequestsGetRequest', response_type_name='ApprovalRequest', supports_download=False)

    def Invalidate(self, request, global_params=None):
        """Invalidates an existing ApprovalRequest. Returns the updated ApprovalRequest. NOTE: This does not deny access to the resource if another request has been made and approved. It only invalidates a single approval. Returns FAILED_PRECONDITION if the request exists but is not in an approved state.

      Args:
        request: (AccessapprovalFoldersApprovalRequestsInvalidateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApprovalRequest) The response message.
      """
        config = self.GetMethodConfig('Invalidate')
        return self._RunMethod(config, request, global_params=global_params)
    Invalidate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/approvalRequests/{approvalRequestsId}:invalidate', http_method='POST', method_id='accessapproval.folders.approvalRequests.invalidate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:invalidate', request_field='invalidateApprovalRequestMessage', request_type_name='AccessapprovalFoldersApprovalRequestsInvalidateRequest', response_type_name='ApprovalRequest', supports_download=False)

    def List(self, request, global_params=None):
        """Lists approval requests associated with a project, folder, or organization. Approval requests can be filtered by state (pending, active, dismissed). The order is reverse chronological.

      Args:
        request: (AccessapprovalFoldersApprovalRequestsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListApprovalRequestsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/approvalRequests', http_method='GET', method_id='accessapproval.folders.approvalRequests.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/approvalRequests', request_field='', request_type_name='AccessapprovalFoldersApprovalRequestsListRequest', response_type_name='ListApprovalRequestsResponse', supports_download=False)