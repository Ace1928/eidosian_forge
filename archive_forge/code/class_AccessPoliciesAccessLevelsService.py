from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accesscontextmanager.v1alpha import accesscontextmanager_v1alpha_messages as messages
class AccessPoliciesAccessLevelsService(base_api.BaseApiService):
    """Service class for the accessPolicies_accessLevels resource."""
    _NAME = 'accessPolicies_accessLevels'

    def __init__(self, client):
        super(AccesscontextmanagerV1alpha.AccessPoliciesAccessLevelsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an access level. The long- running operation from this RPC has a successful status after the access level propagates to long-lasting storage. If access levels contain errors, an error response is returned for the first error encountered.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAccessLevelsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/accessLevels', http_method='POST', method_id='accesscontextmanager.accessPolicies.accessLevels.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/accessLevels', request_field='accessLevel', request_type_name='AccesscontextmanagerAccessPoliciesAccessLevelsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an access level based on the resource name. The long-running operation from this RPC has a successful status after the access level has been removed from long-lasting storage.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAccessLevelsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/accessLevels/{accessLevelsId}', http_method='DELETE', method_id='accesscontextmanager.accessPolicies.accessLevels.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesAccessLevelsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an access level based on the resource name.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAccessLevelsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessLevel) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/accessLevels/{accessLevelsId}', http_method='GET', method_id='accesscontextmanager.accessPolicies.accessLevels.get', ordered_params=['name'], path_params=['name'], query_params=['accessLevelFormat'], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesAccessLevelsGetRequest', response_type_name='AccessLevel', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all access levels for an access policy.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAccessLevelsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAccessLevelsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/accessLevels', http_method='GET', method_id='accesscontextmanager.accessPolicies.accessLevels.list', ordered_params=['parent'], path_params=['parent'], query_params=['accessLevelFormat', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/accessLevels', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesAccessLevelsListRequest', response_type_name='ListAccessLevelsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an access level. The long- running operation from this RPC has a successful status after the changes to the access level propagate to long-lasting storage. If access levels contain errors, an error response is returned for the first error encountered.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAccessLevelsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/accessLevels/{accessLevelsId}', http_method='PATCH', method_id='accesscontextmanager.accessPolicies.accessLevels.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='accessLevel', request_type_name='AccesscontextmanagerAccessPoliciesAccessLevelsPatchRequest', response_type_name='Operation', supports_download=False)

    def ReplaceAll(self, request, global_params=None):
        """Replaces all existing access levels in an access policy with the access levels provided. This is done atomically. The long-running operation from this RPC has a successful status after all replacements propagate to long-lasting storage. If the replacement contains errors, an error response is returned for the first error encountered. Upon error, the replacement is cancelled, and existing access levels are not affected. The Operation.response field contains ReplaceAccessLevelsResponse. Removing access levels contained in existing service perimeters result in an error.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAccessLevelsReplaceAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ReplaceAll')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceAll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/accessLevels:replaceAll', http_method='POST', method_id='accesscontextmanager.accessPolicies.accessLevels.replaceAll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/accessLevels:replaceAll', request_field='replaceAccessLevelsRequest', request_type_name='AccesscontextmanagerAccessPoliciesAccessLevelsReplaceAllRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the IAM permissions that the caller has on the specified Access Context Manager resource. The resource can be an AccessPolicy, AccessLevel, or ServicePerimeter. This method does not support other resources.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAccessLevelsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/accessLevels/{accessLevelsId}:testIamPermissions', http_method='POST', method_id='accesscontextmanager.accessPolicies.accessLevels.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='AccesscontextmanagerAccessPoliciesAccessLevelsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)