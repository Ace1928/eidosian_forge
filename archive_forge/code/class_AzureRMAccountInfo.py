from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
class AzureRMAccountInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict()
        self.results = dict(changed=False, account_info=[])
        super(AzureRMAccountInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, is_ad_resource=False)

    def exec_module(self, **kwargs):
        result = []
        result = self.list_items()
        self.results['account_info'] = result
        return self.results

    def list_items(self):
        results = {}
        try:
            subscription_list_response = list(self.subscription_client.subscriptions.list())
        except Exception as exc:
            self.fail('Failed to list all subscriptions - {0}'.format(str(exc)))
        results['id'] = subscription_list_response[0].subscription_id
        results['tenantId'] = subscription_list_response[0].tenant_id
        results['homeTenantId'] = subscription_list_response[0].tenant_id
        results['name'] = subscription_list_response[0].display_name
        results['state'] = subscription_list_response[0].state
        results['managedByTenants'] = self.get_managed_by_tenants_list(subscription_list_response[0].managed_by_tenants)
        results['environmentName'] = self.azure_auth._cloud_environment.name
        results['user'] = self.get_aduser_info(subscription_list_response[0].tenant_id)
        return results

    def get_managed_by_tenants_list(self, object_list):
        return [dict(tenantId=item.tenant_id) for item in object_list]

    def get_aduser_info(self, tenant_id):
        user = {}
        self.azure_auth_graphrbac = AzureRMAuth(is_ad_resource=True)
        cred = self.azure_auth_graphrbac.azure_credentials
        base_url = self.azure_auth_graphrbac._cloud_environment.endpoints.active_directory_graph_resource_id
        client = GraphRbacManagementClient(cred, tenant_id, base_url)
        try:
            user_info = client.signed_in_user.get()
            user['name'] = user_info.user_principal_name
            user['type'] = user_info.object_type
        except GraphErrorException as e:
            self.fail('failed to get ad user info {0}'.format(str(e)))
        return user