from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NetworkAttachmentsService(base_api.BaseApiService):
    """Service class for the networkAttachments resource."""
    _NAME = 'networkAttachments'

    def __init__(self, client):
        super(ComputeBeta.NetworkAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all NetworkAttachment resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeNetworkAttachmentsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkAttachmentAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkAttachments.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/networkAttachments', request_field='', request_type_name='ComputeNetworkAttachmentsAggregatedListRequest', response_type_name='NetworkAttachmentAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified NetworkAttachment in the given scope.

      Args:
        request: (ComputeNetworkAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.networkAttachments.delete', ordered_params=['project', 'region', 'networkAttachment'], path_params=['networkAttachment', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkAttachments/{networkAttachment}', request_field='', request_type_name='ComputeNetworkAttachmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified NetworkAttachment resource in the given scope.

      Args:
        request: (ComputeNetworkAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkAttachments.get', ordered_params=['project', 'region', 'networkAttachment'], path_params=['networkAttachment', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/networkAttachments/{networkAttachment}', request_field='', request_type_name='ComputeNetworkAttachmentsGetRequest', response_type_name='NetworkAttachment', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeNetworkAttachmentsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkAttachments.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/networkAttachments/{resource}/getIamPolicy', request_field='', request_type_name='ComputeNetworkAttachmentsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a NetworkAttachment in the specified project in the given scope using the parameters that are included in the request.

      Args:
        request: (ComputeNetworkAttachmentsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkAttachments.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkAttachments', request_field='networkAttachment', request_type_name='ComputeNetworkAttachmentsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the NetworkAttachments for a project in the given scope.

      Args:
        request: (ComputeNetworkAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkAttachmentList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkAttachments.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/networkAttachments', request_field='', request_type_name='ComputeNetworkAttachmentsListRequest', response_type_name='NetworkAttachmentList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified NetworkAttachment resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeNetworkAttachmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.networkAttachments.patch', ordered_params=['project', 'region', 'networkAttachment'], path_params=['networkAttachment', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkAttachments/{networkAttachment}', request_field='networkAttachmentResource', request_type_name='ComputeNetworkAttachmentsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeNetworkAttachmentsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkAttachments.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/networkAttachments/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeNetworkAttachmentsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeNetworkAttachmentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkAttachments.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/networkAttachments/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeNetworkAttachmentsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)