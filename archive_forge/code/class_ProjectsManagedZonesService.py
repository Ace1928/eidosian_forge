from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
class ProjectsManagedZonesService(base_api.BaseApiService):
    """Service class for the projects_managedZones resource."""
    _NAME = 'projects_managedZones'

    def __init__(self, client):
        super(DnsV1alpha2.ProjectsManagedZonesService, self).__init__(client)
        self._upload_configs = {}

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DnsProjectsManagedZonesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='dns/v1alpha2/projects/{projectsId}/managedZones/{managedZonesId}:getIamPolicy', http_method='POST', method_id='dns.projects.managedZones.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='dns/v1alpha2/{+resource}:getIamPolicy', request_field='googleIamV1GetIamPolicyRequest', request_type_name='DnsProjectsManagedZonesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (DnsProjectsManagedZonesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='dns/v1alpha2/projects/{projectsId}/managedZones/{managedZonesId}:setIamPolicy', http_method='POST', method_id='dns.projects.managedZones.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='dns/v1alpha2/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='DnsProjectsManagedZonesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this returns an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DnsProjectsManagedZonesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='dns/v1alpha2/projects/{projectsId}/managedZones/{managedZonesId}:testIamPermissions', http_method='POST', method_id='dns.projects.managedZones.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='dns/v1alpha2/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='DnsProjectsManagedZonesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)