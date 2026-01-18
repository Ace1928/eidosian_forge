from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
class AzureRMRecoveryServicesVault(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.location = None
        self.state = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.url = None
        self.status_code = [200, 201, 202, 204]
        self.body = {}
        self.query_parameters = {}
        self.query_parameters['api-version'] = None
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMRecoveryServicesVault, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def get_api_version(self):
        return '2016-06-01'

    def get_url(self):
        if self.state == 'present' or self.state == 'absent':
            return '/subscriptions/' + self.subscription_id + '/resourceGroups/' + self.resource_group + '/providers/Microsoft.RecoveryServices' + '/vaults' + '/' + self.name

    def get_body(self):
        if self.state == 'present':
            return {'properties': {}, 'sku': {'name': 'Standard'}, 'location': self.location}
        else:
            return {}

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        self.query_parameters['api-version'] = self.get_api_version()
        self.url = self.get_url()
        self.body = self.get_body()
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        old_response = self.get_resource()
        changed = False
        if self.state == 'present':
            if old_response is False:
                changed = True
                response = self.create_recovery_service_vault()
            else:
                changed = False
                response = old_response
        if self.state == 'absent':
            changed = True
            response = self.delete_recovery_service_vault()
        self.results['response'] = response
        self.results['changed'] = changed
        return self.results

    def create_recovery_service_vault(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error in creating Azure Recovery Service Vault.')
            self.fail('Error in creating Azure Recovery Service Vault {0}'.format(str(e)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response

    def delete_recovery_service_vault(self):
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to delete Azure Recovery Service Vault.')
            self.fail('Error while deleting Azure Recovery Service Vault: {0}'.format(str(e)))

    def get_resource(self):
        found = False
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            found = True
        except Exception as e:
            self.log('Recovery Service Vault Does not exist.')
        if found is True:
            response = json.loads(response.body())
            return response
        else:
            return False