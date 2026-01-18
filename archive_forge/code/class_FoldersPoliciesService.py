from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages as messages
class FoldersPoliciesService(base_api.BaseApiService):
    """Service class for the folders_policies resource."""
    _NAME = 'folders_policies'

    def __init__(self, client):
        super(OrgpolicyV2.FoldersPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a policy. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the constraint does not exist. Returns a `google.rpc.Status` with `google.rpc.Code.ALREADY_EXISTS` if the policy already exists on the given Google Cloud resource.

      Args:
        request: (OrgpolicyFoldersPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2Policy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/policies', http_method='POST', method_id='orgpolicy.folders.policies.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/policies', request_field='googleCloudOrgpolicyV2Policy', request_type_name='OrgpolicyFoldersPoliciesCreateRequest', response_type_name='GoogleCloudOrgpolicyV2Policy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a policy. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the constraint or organization policy does not exist.

      Args:
        request: (OrgpolicyFoldersPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/policies/{policiesId}', http_method='DELETE', method_id='orgpolicy.folders.policies.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v2/{+name}', request_field='', request_type_name='OrgpolicyFoldersPoliciesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a policy on a resource. If no policy is set on the resource, `NOT_FOUND` is returned. The `etag` value can be used with `UpdatePolicy()` to update a policy during read-modify-write.

      Args:
        request: (OrgpolicyFoldersPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2Policy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/policies/{policiesId}', http_method='GET', method_id='orgpolicy.folders.policies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='OrgpolicyFoldersPoliciesGetRequest', response_type_name='GoogleCloudOrgpolicyV2Policy', supports_download=False)

    def GetEffectivePolicy(self, request, global_params=None):
        """Gets the effective policy on a resource. This is the result of merging policies in the resource hierarchy and evaluating conditions. The returned policy will not have an `etag` or `condition` set because it is an evaluated policy across multiple resources. Subtrees of Resource Manager resource hierarchy with 'under:' prefix will not be expanded.

      Args:
        request: (OrgpolicyFoldersPoliciesGetEffectivePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2Policy) The response message.
      """
        config = self.GetMethodConfig('GetEffectivePolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetEffectivePolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/policies/{policiesId}:getEffectivePolicy', http_method='GET', method_id='orgpolicy.folders.policies.getEffectivePolicy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:getEffectivePolicy', request_field='', request_type_name='OrgpolicyFoldersPoliciesGetEffectivePolicyRequest', response_type_name='GoogleCloudOrgpolicyV2Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves all of the policies that exist on a particular resource.

      Args:
        request: (OrgpolicyFoldersPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2ListPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/policies', http_method='GET', method_id='orgpolicy.folders.policies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/policies', request_field='', request_type_name='OrgpolicyFoldersPoliciesListRequest', response_type_name='GoogleCloudOrgpolicyV2ListPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a policy. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the constraint or the policy do not exist. Returns a `google.rpc.Status` with `google.rpc.Code.ABORTED` if the etag supplied in the request does not match the persisted etag of the policy Note: the supplied policy will perform a full overwrite of all fields.

      Args:
        request: (OrgpolicyFoldersPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2Policy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/policies/{policiesId}', http_method='PATCH', method_id='orgpolicy.folders.policies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudOrgpolicyV2Policy', request_type_name='OrgpolicyFoldersPoliciesPatchRequest', response_type_name='GoogleCloudOrgpolicyV2Policy', supports_download=False)