from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ServiceAttachmentsService(base_api.BaseApiService):
    """Service class for the serviceAttachments resource."""
    _NAME = 'serviceAttachments'

    def __init__(self, client):
        super(ComputeBeta.ServiceAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all ServiceAttachment resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeServiceAttachmentsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAttachmentAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.serviceAttachments.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/serviceAttachments', request_field='', request_type_name='ComputeServiceAttachmentsAggregatedListRequest', response_type_name='ServiceAttachmentAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified ServiceAttachment in the given scope.

      Args:
        request: (ComputeServiceAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.serviceAttachments.delete', ordered_params=['project', 'region', 'serviceAttachment'], path_params=['project', 'region', 'serviceAttachment'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/serviceAttachments/{serviceAttachment}', request_field='', request_type_name='ComputeServiceAttachmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified ServiceAttachment resource in the given scope.

      Args:
        request: (ComputeServiceAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.serviceAttachments.get', ordered_params=['project', 'region', 'serviceAttachment'], path_params=['project', 'region', 'serviceAttachment'], query_params=[], relative_path='projects/{project}/regions/{region}/serviceAttachments/{serviceAttachment}', request_field='', request_type_name='ComputeServiceAttachmentsGetRequest', response_type_name='ServiceAttachment', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeServiceAttachmentsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.serviceAttachments.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/serviceAttachments/{resource}/getIamPolicy', request_field='', request_type_name='ComputeServiceAttachmentsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a ServiceAttachment in the specified project in the given scope using the parameters that are included in the request.

      Args:
        request: (ComputeServiceAttachmentsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.serviceAttachments.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/serviceAttachments', request_field='serviceAttachment', request_type_name='ComputeServiceAttachmentsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the ServiceAttachments for a project in the given scope.

      Args:
        request: (ComputeServiceAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAttachmentList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.serviceAttachments.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/serviceAttachments', request_field='', request_type_name='ComputeServiceAttachmentsListRequest', response_type_name='ServiceAttachmentList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified ServiceAttachment resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeServiceAttachmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.serviceAttachments.patch', ordered_params=['project', 'region', 'serviceAttachment'], path_params=['project', 'region', 'serviceAttachment'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/serviceAttachments/{serviceAttachment}', request_field='serviceAttachmentResource', request_type_name='ComputeServiceAttachmentsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeServiceAttachmentsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.serviceAttachments.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/serviceAttachments/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeServiceAttachmentsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeServiceAttachmentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.serviceAttachments.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/serviceAttachments/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeServiceAttachmentsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)