from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
class AzureRMRecoveryServicesVaultInfo(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True))
        self.resource_group = None
        self.name = None
        self.body = {}
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.url = None
        self.status_code = [200, 201, 202, 204]
        self.query_parameters = {}
        self.query_parameters['api-version'] = None
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMRecoveryServicesVaultInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def get_api_version(self):
        return '2016-06-01'

    def get_url(self):
        return '/subscriptions/' + self.subscription_id + '/resourceGroups/' + self.resource_group + '/providers/Microsoft.RecoveryServices' + '/vaults' + '/' + self.name

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        self.query_parameters['api-version'] = self.get_api_version()
        self.url = self.get_url()
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        changed = True
        response = self.get_recovery_service_vault_info()
        self.results['response'] = response
        self.results['changed'] = changed
        return self.results

    def get_recovery_service_vault_info(self):
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error in fetching Azure Recovery Service Vault Details.')
            self.fail('Error in fetching Azure Recovery Service Vault Details {0}'.format(str(e)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response