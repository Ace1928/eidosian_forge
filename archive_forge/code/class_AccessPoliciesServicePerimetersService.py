from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accesscontextmanager.v1alpha import accesscontextmanager_v1alpha_messages as messages
class AccessPoliciesServicePerimetersService(base_api.BaseApiService):
    """Service class for the accessPolicies_servicePerimeters resource."""
    _NAME = 'accessPolicies_servicePerimeters'

    def __init__(self, client):
        super(AccesscontextmanagerV1alpha.AccessPoliciesServicePerimetersService, self).__init__(client)
        self._upload_configs = {}

    def Commit(self, request, global_params=None):
        """Commits the dry-run specification for all the service perimeters in an access policy. A commit operation on a service perimeter involves copying its `spec` field to the `status` field of the service perimeter. Only service perimeters with `use_explicit_dry_run_spec` field set to true are affected by a commit operation. The long-running operation from this RPC has a successful status after the dry-run specifications for all the service perimeters have been committed. If a commit fails, it causes the long-running operation to return an error response and the entire commit operation is cancelled. When successful, the Operation.response field contains CommitServicePerimetersResponse. The `dry_run` and the `spec` fields are cleared after a successful commit operation.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersCommitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Commit')
        return self._RunMethod(config, request, global_params=global_params)
    Commit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters:commit', http_method='POST', method_id='accesscontextmanager.accessPolicies.servicePerimeters.commit', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/servicePerimeters:commit', request_field='commitServicePerimetersRequest', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersCommitRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a service perimeter. The long-running operation from this RPC has a successful status after the service perimeter propagates to long-lasting storage. If a service perimeter contains errors, an error response is returned for the first error encountered.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters', http_method='POST', method_id='accesscontextmanager.accessPolicies.servicePerimeters.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/servicePerimeters', request_field='servicePerimeter', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a service perimeter based on the resource name. The long-running operation from this RPC has a successful status after the service perimeter is removed from long-lasting storage.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters/{servicePerimetersId}', http_method='DELETE', method_id='accesscontextmanager.accessPolicies.servicePerimeters.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a service perimeter based on the resource name.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServicePerimeter) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters/{servicePerimetersId}', http_method='GET', method_id='accesscontextmanager.accessPolicies.servicePerimeters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersGetRequest', response_type_name='ServicePerimeter', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all service perimeters for an access policy.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServicePerimetersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters', http_method='GET', method_id='accesscontextmanager.accessPolicies.servicePerimeters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/servicePerimeters', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersListRequest', response_type_name='ListServicePerimetersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a service perimeter. The long-running operation from this RPC has a successful status after the service perimeter propagates to long-lasting storage. If a service perimeter contains errors, an error response is returned for the first error encountered.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters/{servicePerimetersId}', http_method='PATCH', method_id='accesscontextmanager.accessPolicies.servicePerimeters.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='servicePerimeter', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersPatchRequest', response_type_name='Operation', supports_download=False)

    def ReplaceAll(self, request, global_params=None):
        """Replace all existing service perimeters in an access policy with the service perimeters provided. This is done atomically. The long-running operation from this RPC has a successful status after all replacements propagate to long-lasting storage. Replacements containing errors result in an error response for the first error encountered. Upon an error, replacement are cancelled and existing service perimeters are not affected. The Operation.response field contains ReplaceServicePerimetersResponse.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersReplaceAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ReplaceAll')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceAll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters:replaceAll', http_method='POST', method_id='accesscontextmanager.accessPolicies.servicePerimeters.replaceAll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/servicePerimeters:replaceAll', request_field='replaceServicePerimetersRequest', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersReplaceAllRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the IAM permissions that the caller has on the specified Access Context Manager resource. The resource can be an AccessPolicy, AccessLevel, or ServicePerimeter. This method does not support other resources.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/servicePerimeters/{servicePerimetersId}:testIamPermissions', http_method='POST', method_id='accesscontextmanager.accessPolicies.servicePerimeters.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='AccesscontextmanagerAccessPoliciesServicePerimetersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)