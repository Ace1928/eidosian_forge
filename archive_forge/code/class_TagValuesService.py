from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v3 import cloudresourcemanager_v3_messages as messages
class TagValuesService(base_api.BaseApiService):
    """Service class for the tagValues resource."""
    _NAME = 'tagValues'

    def __init__(self, client):
        super(CloudresourcemanagerV3.TagValuesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a TagValue as a child of the specified TagKey. If a another request with the same parameters is sent while the original request is in process the second request will receive an error. A maximum of 1000 TagValues can exist under a TagKey at any given time.

      Args:
        request: (CloudresourcemanagerTagValuesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudresourcemanager.tagValues.create', ordered_params=[], path_params=[], query_params=['validateOnly'], relative_path='v3/tagValues', request_field='tagValue', request_type_name='CloudresourcemanagerTagValuesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TagValue. The TagValue cannot have any bindings when it is deleted.

      Args:
        request: (CloudresourcemanagerTagValuesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}', http_method='DELETE', method_id='cloudresourcemanager.tagValues.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v3/{+name}', request_field='', request_type_name='CloudresourcemanagerTagValuesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a TagValue. This method will return `PERMISSION_DENIED` if the value does not exist or the user does not have permission to view it.

      Args:
        request: (CloudresourcemanagerTagValuesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TagValue) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}', http_method='GET', method_id='cloudresourcemanager.tagValues.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='CloudresourcemanagerTagValuesGetRequest', response_type_name='TagValue', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a TagValue. The returned policy may be empty if no such policy or resource exists. The `resource` field should be the TagValue's resource name. For example: `tagValues/1234`. The caller must have the `cloudresourcemanager.googleapis.com/tagValues.getIamPolicy` permission on the identified TagValue to get the access control policy.

      Args:
        request: (CloudresourcemanagerTagValuesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}:getIamPolicy', http_method='POST', method_id='cloudresourcemanager.tagValues.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v3/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='CloudresourcemanagerTagValuesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetNamespaced(self, request, global_params=None):
        """Retrieves a TagValue by its namespaced name. This method will return `PERMISSION_DENIED` if the value does not exist or the user does not have permission to view it.

      Args:
        request: (CloudresourcemanagerTagValuesGetNamespacedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TagValue) The response message.
      """
        config = self.GetMethodConfig('GetNamespaced')
        return self._RunMethod(config, request, global_params=global_params)
    GetNamespaced.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.tagValues.getNamespaced', ordered_params=[], path_params=[], query_params=['name'], relative_path='v3/tagValues/namespaced', request_field='', request_type_name='CloudresourcemanagerTagValuesGetNamespacedRequest', response_type_name='TagValue', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all TagValues for a specific TagKey.

      Args:
        request: (CloudresourcemanagerTagValuesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTagValuesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.tagValues.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v3/tagValues', request_field='', request_type_name='CloudresourcemanagerTagValuesListRequest', response_type_name='ListTagValuesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the attributes of the TagValue resource.

      Args:
        request: (CloudresourcemanagerTagValuesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}', http_method='PATCH', method_id='cloudresourcemanager.tagValues.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v3/{+name}', request_field='tagValue', request_type_name='CloudresourcemanagerTagValuesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on a TagValue, replacing any existing policy. The `resource` field should be the TagValue's resource name. For example: `tagValues/1234`. The caller must have `resourcemanager.tagValues.setIamPolicy` permission on the identified tagValue.

      Args:
        request: (CloudresourcemanagerTagValuesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}:setIamPolicy', http_method='POST', method_id='cloudresourcemanager.tagValues.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v3/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CloudresourcemanagerTagValuesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified TagValue. The `resource` field should be the TagValue's resource name. For example: `tagValues/1234`. There are no permissions required for making this API call.

      Args:
        request: (CloudresourcemanagerTagValuesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagValues/{tagValuesId}:testIamPermissions', http_method='POST', method_id='cloudresourcemanager.tagValues.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v3/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CloudresourcemanagerTagValuesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)