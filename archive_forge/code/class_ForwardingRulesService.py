from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ForwardingRulesService(base_api.BaseApiService):
    """Service class for the forwardingRules resource."""
    _NAME = 'forwardingRules'

    def __init__(self, client):
        super(ComputeBeta.ForwardingRulesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of forwarding rules. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeForwardingRulesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ForwardingRuleAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.forwardingRules.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/forwardingRules', request_field='', request_type_name='ComputeForwardingRulesAggregatedListRequest', response_type_name='ForwardingRuleAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified ForwardingRule resource.

      Args:
        request: (ComputeForwardingRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.forwardingRules.delete', ordered_params=['project', 'region', 'forwardingRule'], path_params=['forwardingRule', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/forwardingRules/{forwardingRule}', request_field='', request_type_name='ComputeForwardingRulesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified ForwardingRule resource.

      Args:
        request: (ComputeForwardingRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ForwardingRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.forwardingRules.get', ordered_params=['project', 'region', 'forwardingRule'], path_params=['forwardingRule', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/forwardingRules/{forwardingRule}', request_field='', request_type_name='ComputeForwardingRulesGetRequest', response_type_name='ForwardingRule', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a ForwardingRule resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeForwardingRulesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.forwardingRules.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/forwardingRules', request_field='forwardingRule', request_type_name='ComputeForwardingRulesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of ForwardingRule resources available to the specified project and region.

      Args:
        request: (ComputeForwardingRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ForwardingRuleList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.forwardingRules.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/forwardingRules', request_field='', request_type_name='ComputeForwardingRulesListRequest', response_type_name='ForwardingRuleList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified forwarding rule with the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules. Currently, you can only patch the network_tier field.

      Args:
        request: (ComputeForwardingRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.forwardingRules.patch', ordered_params=['project', 'region', 'forwardingRule'], path_params=['forwardingRule', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/forwardingRules/{forwardingRule}', request_field='forwardingRuleResource', request_type_name='ComputeForwardingRulesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on the specified resource. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeForwardingRulesSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.forwardingRules.setLabels', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/forwardingRules/{resource}/setLabels', request_field='regionSetLabelsRequest', request_type_name='ComputeForwardingRulesSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def SetTarget(self, request, global_params=None):
        """Changes target URL for forwarding rule. The new target should be of the same type as the old target.

      Args:
        request: (ComputeForwardingRulesSetTargetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetTarget')
        return self._RunMethod(config, request, global_params=global_params)
    SetTarget.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.forwardingRules.setTarget', ordered_params=['project', 'region', 'forwardingRule'], path_params=['forwardingRule', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/forwardingRules/{forwardingRule}/setTarget', request_field='targetReference', request_type_name='ComputeForwardingRulesSetTargetRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeForwardingRulesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.forwardingRules.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/forwardingRules/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeForwardingRulesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)