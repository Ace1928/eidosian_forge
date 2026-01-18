from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v1 import cloudresourcemanager_v1_messages as messages
def GetOrgPolicy(self, request, global_params=None):
    """Gets a `Policy` on a resource. If no `Policy` is set on the resource, a `Policy` is returned with default values including `POLICY_TYPE_NOT_SET` for the `policy_type oneof`. The `etag` value can be used with `SetOrgPolicy()` to create or update a `Policy` during read-modify-write.

      Args:
        request: (CloudresourcemanagerProjectsGetOrgPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrgPolicy) The response message.
      """
    config = self.GetMethodConfig('GetOrgPolicy')
    return self._RunMethod(config, request, global_params=global_params)