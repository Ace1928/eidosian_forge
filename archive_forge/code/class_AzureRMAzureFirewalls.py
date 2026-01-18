from __future__ import absolute_import, division, print_function
import time
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMAzureFirewalls(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', disposition='resource_group_name', required=True), name=dict(type='str', disposition='azure_firewall_name', required=True), location=dict(type='str', updatable=False, disposition='/', comparison='location'), application_rule_collections=dict(type='list', elements='dict', disposition='/properties/applicationRuleCollections', options=dict(priority=dict(type='int', disposition='properties/*'), action=dict(type='str', choices=['allow', 'deny'], disposition='properties/action/type', pattern='camelize'), rules=dict(type='list', elements='raw', disposition='properties/*', options=dict(name=dict(type='str'), description=dict(type='str'), source_addresses=dict(type='list', elements='str', disposition='sourceAddresses'), protocols=dict(type='list', elements='dict', options=dict(type=dict(type='str', disposition='protocolType'), port=dict(type='str'))), target_fqdns=dict(type='list', elements='raw', disposition='targetFqdns'), fqdn_tags=dict(type='list', elements='raw', disposition='fqdnTags'))), name=dict(type='str'))), nat_rule_collections=dict(type='list', elements='dict', disposition='/properties/natRuleCollections', options=dict(priority=dict(type='int', disposition='properties/*'), action=dict(type='str', disposition='properties/action/type', choices=['snat', 'dnat'], pattern='camelize'), rules=dict(type='list', elements='dict', disposition='properties/*', options=dict(name=dict(type='str'), description=dict(type='str'), source_addresses=dict(type='list', elements='str', disposition='sourceAddresses'), destination_addresses=dict(type='list', elements='str', disposition='destinationAddresses'), destination_ports=dict(type='list', elements='str', disposition='destinationPorts'), protocols=dict(type='list', elements='raw'), translated_address=dict(type='str', disposition='translatedAddress'), translated_port=dict(type='str', disposition='translatedPort'))), name=dict(type='str'))), network_rule_collections=dict(type='list', elements='dict', disposition='/properties/networkRuleCollections', options=dict(priority=dict(type='int', disposition='properties/*'), action=dict(type='str', choices=['allow', 'deny'], disposition='properties/action/type', pattern='camelize'), rules=dict(type='list', elements='dict', disposition='properties/*', options=dict(name=dict(type='str'), description=dict(type='str'), protocols=dict(type='list', elements='raw'), source_addresses=dict(type='list', elements='str', disposition='sourceAddresses'), destination_addresses=dict(type='list', elements='str', disposition='destinationAddresses'), destination_ports=dict(type='list', elements='str', disposition='destinationPorts'))), name=dict(type='str'))), ip_configurations=dict(type='list', elements='dict', disposition='/properties/ipConfigurations', options=dict(subnet=dict(type='raw', disposition='properties/subnet/id', pattern='/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/virtualNetworks/{virtual_network_name}/subnets/{name}'), public_ip_address=dict(type='raw', disposition='properties/publicIPAddress/id', pattern='/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/publicIPAddresses/{name}'), name=dict(type='str'))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.body = {}
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200, 201, 202]
        self.to_do = Actions.NoAction
        self.body = {}
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2018-11-01'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMAzureFirewalls, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.body:
            self.body['location'] = resource_group.location
        self.url = '/subscriptions' + '/' + self.subscription_id + '/resourceGroups' + '/' + self.resource_group + '/providers' + '/Microsoft.Network' + '/azureFirewalls' + '/' + self.name
        old_response = self.get_resource()
        if not old_response:
            self.log("AzureFirewall instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('AzureFirewall instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            else:
                modifiers = {}
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                self.results['modifiers'] = modifiers
                self.results['compare'] = []
                if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the AzureFirewall instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_resource()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('AzureFirewall instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
            while self.get_resource():
                time.sleep(20)
        else:
            self.log('AzureFirewall instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            while response['properties']['provisioningState'] == 'Updating':
                time.sleep(30)
                response = self.get_resource()
        return self.results

    def create_update_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as exc:
            self.log('Error attempting to create the AzureFirewall instance.')
            self.fail('Error creating the AzureFirewall instance: {0}'.format(str(exc)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response

    def delete_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to delete the AzureFirewall instance.')
            self.fail('Error deleting the AzureFirewall instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        found = False
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            response = json.loads(response.body())
            found = True
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Did not find the AzureFirewall instance.')
        if found is True:
            return response
        return False