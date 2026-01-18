from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
class BackupAzureVM(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), recovery_vault_name=dict(type='str', required=True), resource_id=dict(type='str', required=True), backup_policy_id=dict(type='str', required=True), recovery_point_expiry_time=dict(type='str'), state=dict(type='str', default='create', choices=['create', 'update', 'delete', 'stop', 'backup']))
        self.resource_group = None
        self.recovery_vault_name = None
        self.resource_id = None
        self.backup_policy_id = None
        self.recovery_point_expiry_time = None
        self.state = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.url = None
        self.status_code = [200, 201, 202, 204]
        self.to_do = Actions.NoAction
        self.body = {}
        self.query_parameters = {}
        self.query_parameters['api-version'] = None
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(BackupAzureVM, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def get_api_version(self):
        return '2019-05-13' if self.state == 'create' or self.state == 'update' or self.state == 'delete' or (self.state == 'stop') else '2016-12-01'

    def get_url(self):
        sub_id = self.subscription_id
        if self.module.params.get('subscription_id'):
            sub_id = self.module.params.get('subscription_id')
        if self.state == 'create' or self.state == 'update' or self.state == 'delete' or (self.state == 'stop'):
            return '/subscriptions' + '/' + sub_id + '/resourceGroups' + '/' + self.resource_group + '/providers' + '/Microsoft.RecoveryServices' + '/vaults' + '/' + self.recovery_vault_name + '/backupFabrics/Azure/protectionContainers/' + 'iaasvmcontainer;iaasvmcontainerv2;' + self.parse_resource_to_dict(self.resource_id)['resource_group'] + ';' + self.parse_resource_to_dict(self.resource_id)['name'] + '/protectedItems/' + 'vm;iaasvmcontainerv2;' + self.parse_resource_to_dict(self.resource_id)['resource_group'] + ';' + self.parse_resource_to_dict(self.resource_id)['name']
        if self.state == 'backup':
            return '/subscriptions' + '/' + sub_id + '/resourceGroups' + '/' + self.resource_group + '/providers' + '/Microsoft.RecoveryServices' + '/vaults' + '/' + self.recovery_vault_name + '/backupFabrics/Azure/protectionContainers/' + 'iaasvmcontainer;iaasvmcontainerv2;' + self.parse_resource_to_dict(self.resource_id)['resource_group'] + ';' + self.parse_resource_to_dict(self.resource_id)['name'] + '/protectedItems/' + 'vm;iaasvmcontainerv2;' + self.parse_resource_to_dict(self.resource_id)['resource_group'] + ';' + self.parse_resource_to_dict(self.resource_id)['name'] + '/backup'

    def get_body(self):
        if self.state == 'create' or self.state == 'update':
            return {'properties': {'protectedItemType': 'Microsoft.Compute/virtualMachines', 'sourceResourceId': self.resource_id, 'policyId': self.backup_policy_id}}
        elif self.state == 'backup':
            body = {'properties': {'objectType': 'IaasVMBackupRequest'}}
            if self.recovery_point_expiry_time:
                body['properties']['recoveryPointExpiryTimeInUTC'] = self.recovery_point_expiry_time
            return body
        elif self.state == 'stop':
            return {'properties': {'protectedItemType': 'Microsoft.Compute/virtualMachines', 'sourceResourceId': self.resource_id, 'protectionState': 'ProtectionStopped'}}
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
        changed = False
        if self.state == 'create' or self.state == 'update':
            changed = True
            response = self.enable_update_protection_for_azure_vm()
        if self.state == 'delete':
            changed = True
            response = self.stop_protection_and_delete_data()
        if self.state == 'stop':
            changed = True
            response = self.stop_protection_but_retain_existing_data()
        if self.state == 'backup':
            changed = True
            response = self.trigger_on_demand_backup()
        self.results['response'] = response
        self.results['changed'] = changed
        return self.results

    def enable_update_protection_for_azure_vm(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error in enabling/updating protection for Azure VM.')
            self.fail('Error in creating/updating protection for Azure VM {0}'.format(str(e)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response

    def stop_protection_but_retain_existing_data(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to stop protection.')
            self.fail('Error in disabling the protection: {0}'.format(str(e)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response

    def stop_protection_and_delete_data(self):
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to delete backup.')
            self.fail('Error deleting the azure backup: {0}'.format(str(e)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response

    def trigger_on_demand_backup(self):
        try:
            response = self.mgmt_client.query(self.url, 'POST', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to backup azure vm.')
            self.fail('Error while taking on-demand backup: {0}'.format(str(e)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response