from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class LocationsWorkforcePoolsService(base_api.BaseApiService):
    """Service class for the locations_workforcePools resource."""
    _NAME = 'locations_workforcePools'

    def __init__(self, client):
        super(IamV1.LocationsWorkforcePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkforcePool. You cannot reuse the name of a deleted pool until 30 days after deletion.

      Args:
        request: (IamLocationsWorkforcePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools', http_method='POST', method_id='iam.locations.workforcePools.create', ordered_params=['location'], path_params=['location'], query_params=['workforcePoolId'], relative_path='v1/{+location}/workforcePools', request_field='workforcePool', request_type_name='IamLocationsWorkforcePoolsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkforcePool. You cannot use a deleted WorkforcePool to exchange external credentials for Google Cloud credentials. However, deletion does not revoke credentials that have already been issued. Credentials issued for a deleted pool do not grant access to resources. If the pool is undeleted, and the credentials are not expired, they grant access again. You can undelete a pool for 30 days. After 30 days, deletion is permanent. You cannot update deleted pools. However, you can view and list them.

      Args:
        request: (IamLocationsWorkforcePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}', http_method='DELETE', method_id='iam.locations.workforcePools.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkforcePool.

      Args:
        request: (IamLocationsWorkforcePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}', http_method='GET', method_id='iam.locations.workforcePools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsGetRequest', response_type_name='WorkforcePool', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets IAM policies on a WorkforcePool.

      Args:
        request: (IamLocationsWorkforcePoolsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}:getIamPolicy', http_method='POST', method_id='iam.locations.workforcePools.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='IamLocationsWorkforcePoolsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkforcePools under the specified parent. If `show_deleted` is set to `true`, then deleted pools are also listed.

      Args:
        request: (IamLocationsWorkforcePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkforcePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools', http_method='GET', method_id='iam.locations.workforcePools.list', ordered_params=['location'], path_params=['location'], query_params=['pageSize', 'pageToken', 'parent', 'showDeleted'], relative_path='v1/{+location}/workforcePools', request_field='', request_type_name='IamLocationsWorkforcePoolsListRequest', response_type_name='ListWorkforcePoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkforcePool.

      Args:
        request: (IamLocationsWorkforcePoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}', http_method='PATCH', method_id='iam.locations.workforcePools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='workforcePool', request_type_name='IamLocationsWorkforcePoolsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets IAM policies on a WorkforcePool.

      Args:
        request: (IamLocationsWorkforcePoolsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}:setIamPolicy', http_method='POST', method_id='iam.locations.workforcePools.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='IamLocationsWorkforcePoolsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the caller's permissions on the WorkforcePool. If the pool does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error.

      Args:
        request: (IamLocationsWorkforcePoolsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}:testIamPermissions', http_method='POST', method_id='iam.locations.workforcePools.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='IamLocationsWorkforcePoolsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkforcePool, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamLocationsWorkforcePoolsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}:undelete', http_method='POST', method_id='iam.locations.workforcePools.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkforcePoolRequest', request_type_name='IamLocationsWorkforcePoolsUndeleteRequest', response_type_name='Operation', supports_download=False)