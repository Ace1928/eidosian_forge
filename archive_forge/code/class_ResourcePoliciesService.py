from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ResourcePoliciesService(base_api.BaseApiService):
    """Service class for the resourcePolicies resource."""
    _NAME = 'resourcePolicies'

    def __init__(self, client):
        super(ComputeBeta.ResourcePoliciesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of resource policies. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeResourcePoliciesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourcePolicyAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.resourcePolicies.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/resourcePolicies', request_field='', request_type_name='ComputeResourcePoliciesAggregatedListRequest', response_type_name='ResourcePolicyAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified resource policy.

      Args:
        request: (ComputeResourcePoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.resourcePolicies.delete', ordered_params=['project', 'region', 'resourcePolicy'], path_params=['project', 'region', 'resourcePolicy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/resourcePolicies/{resourcePolicy}', request_field='', request_type_name='ComputeResourcePoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves all information of the specified resource policy.

      Args:
        request: (ComputeResourcePoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourcePolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.resourcePolicies.get', ordered_params=['project', 'region', 'resourcePolicy'], path_params=['project', 'region', 'resourcePolicy'], query_params=[], relative_path='projects/{project}/regions/{region}/resourcePolicies/{resourcePolicy}', request_field='', request_type_name='ComputeResourcePoliciesGetRequest', response_type_name='ResourcePolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeResourcePoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.resourcePolicies.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/resourcePolicies/{resource}/getIamPolicy', request_field='', request_type_name='ComputeResourcePoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new resource policy.

      Args:
        request: (ComputeResourcePoliciesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.resourcePolicies.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/resourcePolicies', request_field='resourcePolicy', request_type_name='ComputeResourcePoliciesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """A list all the resource policies that have been configured for the specified project in specified region.

      Args:
        request: (ComputeResourcePoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourcePolicyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.resourcePolicies.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/resourcePolicies', request_field='', request_type_name='ComputeResourcePoliciesListRequest', response_type_name='ResourcePolicyList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Modify the specified resource policy.

      Args:
        request: (ComputeResourcePoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.resourcePolicies.patch', ordered_params=['project', 'region', 'resourcePolicy'], path_params=['project', 'region', 'resourcePolicy'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/regions/{region}/resourcePolicies/{resourcePolicy}', request_field='resourcePolicyResource', request_type_name='ComputeResourcePoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeResourcePoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.resourcePolicies.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/resourcePolicies/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeResourcePoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeResourcePoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.resourcePolicies.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/resourcePolicies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeResourcePoliciesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)