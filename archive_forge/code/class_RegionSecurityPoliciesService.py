from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionSecurityPoliciesService(base_api.BaseApiService):
    """Service class for the regionSecurityPolicies resource."""
    _NAME = 'regionSecurityPolicies'

    def __init__(self, client):
        super(ComputeBeta.RegionSecurityPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def AddRule(self, request, global_params=None):
        """Inserts a rule into a security policy.

      Args:
        request: (ComputeRegionSecurityPoliciesAddRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddRule')
        return self._RunMethod(config, request, global_params=global_params)
    AddRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSecurityPolicies.addRule', ordered_params=['project', 'region', 'securityPolicy'], path_params=['project', 'region', 'securityPolicy'], query_params=['validateOnly'], relative_path='projects/{project}/regions/{region}/securityPolicies/{securityPolicy}/addRule', request_field='securityPolicyRule', request_type_name='ComputeRegionSecurityPoliciesAddRuleRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified policy.

      Args:
        request: (ComputeRegionSecurityPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionSecurityPolicies.delete', ordered_params=['project', 'region', 'securityPolicy'], path_params=['project', 'region', 'securityPolicy'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/securityPolicies/{securityPolicy}', request_field='', request_type_name='ComputeRegionSecurityPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """List all of the ordered rules present in a single specified policy.

      Args:
        request: (ComputeRegionSecurityPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSecurityPolicies.get', ordered_params=['project', 'region', 'securityPolicy'], path_params=['project', 'region', 'securityPolicy'], query_params=[], relative_path='projects/{project}/regions/{region}/securityPolicies/{securityPolicy}', request_field='', request_type_name='ComputeRegionSecurityPoliciesGetRequest', response_type_name='SecurityPolicy', supports_download=False)

    def GetRule(self, request, global_params=None):
        """Gets a rule at the specified priority.

      Args:
        request: (ComputeRegionSecurityPoliciesGetRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyRule) The response message.
      """
        config = self.GetMethodConfig('GetRule')
        return self._RunMethod(config, request, global_params=global_params)
    GetRule.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSecurityPolicies.getRule', ordered_params=['project', 'region', 'securityPolicy'], path_params=['project', 'region', 'securityPolicy'], query_params=['priority'], relative_path='projects/{project}/regions/{region}/securityPolicies/{securityPolicy}/getRule', request_field='', request_type_name='ComputeRegionSecurityPoliciesGetRuleRequest', response_type_name='SecurityPolicyRule', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new policy in the specified project using the data included in the request.

      Args:
        request: (ComputeRegionSecurityPoliciesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSecurityPolicies.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId', 'validateOnly'], relative_path='projects/{project}/regions/{region}/securityPolicies', request_field='securityPolicy', request_type_name='ComputeRegionSecurityPoliciesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """List all the policies that have been configured for the specified project and region.

      Args:
        request: (ComputeRegionSecurityPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionSecurityPolicies.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/securityPolicies', request_field='', request_type_name='ComputeRegionSecurityPoliciesListRequest', response_type_name='SecurityPolicyList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified policy with the data included in the request. To clear fields in the policy, leave the fields empty and specify them in the updateMask. This cannot be used to be update the rules in the policy. Please use the per rule methods like addRule, patchRule, and removeRule instead.

      Args:
        request: (ComputeRegionSecurityPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionSecurityPolicies.patch', ordered_params=['project', 'region', 'securityPolicy'], path_params=['project', 'region', 'securityPolicy'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/regions/{region}/securityPolicies/{securityPolicy}', request_field='securityPolicyResource', request_type_name='ComputeRegionSecurityPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def PatchRule(self, request, global_params=None):
        """Patches a rule at the specified priority. To clear fields in the rule, leave the fields empty and specify them in the updateMask.

      Args:
        request: (ComputeRegionSecurityPoliciesPatchRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchRule')
        return self._RunMethod(config, request, global_params=global_params)
    PatchRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSecurityPolicies.patchRule', ordered_params=['project', 'region', 'securityPolicy'], path_params=['project', 'region', 'securityPolicy'], query_params=['priority', 'updateMask', 'validateOnly'], relative_path='projects/{project}/regions/{region}/securityPolicies/{securityPolicy}/patchRule', request_field='securityPolicyRule', request_type_name='ComputeRegionSecurityPoliciesPatchRuleRequest', response_type_name='Operation', supports_download=False)

    def RemoveRule(self, request, global_params=None):
        """Deletes a rule at the specified priority.

      Args:
        request: (ComputeRegionSecurityPoliciesRemoveRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveRule')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionSecurityPolicies.removeRule', ordered_params=['project', 'region', 'securityPolicy'], path_params=['project', 'region', 'securityPolicy'], query_params=['priority'], relative_path='projects/{project}/regions/{region}/securityPolicies/{securityPolicy}/removeRule', request_field='', request_type_name='ComputeRegionSecurityPoliciesRemoveRuleRequest', response_type_name='Operation', supports_download=False)