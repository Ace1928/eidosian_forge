from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsFirewallIngressRulesService(base_api.BaseApiService):
    """Service class for the apps_firewall_ingressRules resource."""
    _NAME = 'apps_firewall_ingressRules'

    def __init__(self, client):
        super(AppengineV1beta.AppsFirewallIngressRulesService, self).__init__(client)
        self._upload_configs = {}

    def BatchUpdate(self, request, global_params=None):
        """Replaces the entire firewall ruleset in one bulk operation. This overrides and replaces the rules of an existing firewall with the new rules.If the final rule does not match traffic with the '*' wildcard IP range, then an "allow all" rule is explicitly added to the end of the list.

      Args:
        request: (AppengineAppsFirewallIngressRulesBatchUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchUpdateIngressRulesResponse) The response message.
      """
        config = self.GetMethodConfig('BatchUpdate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchUpdate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/firewall/ingressRules:batchUpdate', http_method='POST', method_id='appengine.apps.firewall.ingressRules.batchUpdate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:batchUpdate', request_field='batchUpdateIngressRulesRequest', request_type_name='AppengineAppsFirewallIngressRulesBatchUpdateRequest', response_type_name='BatchUpdateIngressRulesResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a firewall rule for the application.

      Args:
        request: (AppengineAppsFirewallIngressRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallRule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/firewall/ingressRules', http_method='POST', method_id='appengine.apps.firewall.ingressRules.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}/firewall/ingressRules', request_field='firewallRule', request_type_name='AppengineAppsFirewallIngressRulesCreateRequest', response_type_name='FirewallRule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified firewall rule.

      Args:
        request: (AppengineAppsFirewallIngressRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/firewall/ingressRules/{ingressRulesId}', http_method='DELETE', method_id='appengine.apps.firewall.ingressRules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsFirewallIngressRulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified firewall rule.

      Args:
        request: (AppengineAppsFirewallIngressRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/firewall/ingressRules/{ingressRulesId}', http_method='GET', method_id='appengine.apps.firewall.ingressRules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsFirewallIngressRulesGetRequest', response_type_name='FirewallRule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the firewall rules of an application.

      Args:
        request: (AppengineAppsFirewallIngressRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListIngressRulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/firewall/ingressRules', http_method='GET', method_id='appengine.apps.firewall.ingressRules.list', ordered_params=['parent'], path_params=['parent'], query_params=['matchingAddress', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/firewall/ingressRules', request_field='', request_type_name='AppengineAppsFirewallIngressRulesListRequest', response_type_name='ListIngressRulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified firewall rule.

      Args:
        request: (AppengineAppsFirewallIngressRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallRule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/firewall/ingressRules/{ingressRulesId}', http_method='PATCH', method_id='appengine.apps.firewall.ingressRules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='firewallRule', request_type_name='AppengineAppsFirewallIngressRulesPatchRequest', response_type_name='FirewallRule', supports_download=False)