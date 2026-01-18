from __future__ import absolute_import, division, print_function
import traceback
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
class NetAppCloudManagerConnectorAzure(object):
    """ object initialize and class methods """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.cloudmanager_host_argument_spec()
        self.argument_spec.update(dict(name=dict(required=True, type='str'), state=dict(required=False, choices=['present', 'absent'], default='present'), virtual_machine_size=dict(required=False, type='str', default='Standard_DS3_v2'), resource_group=dict(required=True, type='str'), subscription_id=dict(required=True, type='str'), subnet_name=dict(required=True, type='str', aliases=['subnet_id']), vnet_name=dict(required=True, type='str', aliases=['vnet_id']), vnet_resource_group=dict(required=False, type='str'), location=dict(required=True, type='str'), network_security_resource_group=dict(required=False, type='str'), network_security_group_name=dict(required=True, type='str'), client_id=dict(required=False, type='str'), company=dict(required=True, type='str'), proxy_certificates=dict(required=False, type='list', elements='str'), associate_public_ip_address=dict(required=False, type='bool', default=True), account_id=dict(required=True, type='str'), proxy_url=dict(required=False, type='str'), proxy_user_name=dict(required=False, type='str'), proxy_password=dict(required=False, type='str', no_log=True), admin_username=dict(required=True, type='str'), admin_password=dict(required=True, type='str', no_log=True), storage_account=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[['state', 'absent', ['client_id']]], required_one_of=[['refresh_token', 'sa_client_id']], required_together=[['sa_client_id', 'sa_secret_key']], supports_check_mode=True)
        if HAS_AZURE_LIB is False:
            self.module.fail_json(msg='the python AZURE library azure.mgmt and azure.common is required. Command is pip install azure-mgmt, azure-common. Import error: %s' % str(IMPORT_EXCEPTION))
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        if 'storage_account' not in self.parameters or self.parameters['storage_account'] == '':
            self.parameters['storage_account'] = self.parameters['name'].lower() + 'sa'
        self.rest_api = CloudManagerRestAPI(self.module)

    def get_deploy_azure_vm(self):
        """
        Get Cloud Manager connector for AZURE
        :return:
            Dictionary of current details if Cloud Manager connector for AZURE
            None if Cloud Manager connector for AZURE is not found
        """
        exists = False
        resource_client = get_client_from_cli_profile(ResourceManagementClient)
        try:
            exists = resource_client.deployments.check_existence(self.parameters['resource_group'], self.parameters['name'])
        except CloudError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        if not exists:
            return None
        return exists

    def deploy_azure(self):
        """
        Create Cloud Manager connector for Azure
        :return: client_id
        """
        user_data, client_id = self.register_agent_to_service()
        template = json.loads(self.na_helper.call_template())
        params = json.loads(self.na_helper.call_parameters())
        params['adminUsername']['value'] = self.parameters['admin_username']
        params['adminPassword']['value'] = self.parameters['admin_password']
        params['customData']['value'] = json.dumps(user_data)
        params['location']['value'] = self.parameters['location']
        params['virtualMachineName']['value'] = self.parameters['name']
        params['storageAccount']['value'] = self.parameters['storage_account']
        if self.rest_api.environment == 'stage':
            params['environment']['value'] = self.rest_api.environment
        if '/subscriptions' in self.parameters['vnet_name']:
            network = self.parameters['vnet_name']
        elif self.parameters.get('vnet_resource_group') is not None:
            network = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s' % (self.parameters['subscription_id'], self.parameters['vnet_resource_group'], self.parameters['vnet_name'])
        else:
            network = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s' % (self.parameters['subscription_id'], self.parameters['resource_group'], self.parameters['vnet_name'])
        if '/subscriptions' in self.parameters['subnet_name']:
            subnet = self.parameters['subnet_name']
        else:
            subnet = '%s/subnets/%s' % (network, self.parameters['subnet_name'])
        if self.parameters.get('network_security_resource_group') is not None:
            network_security_group_name = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkSecurityGroups/%s' % (self.parameters['subscription_id'], self.parameters['network_security_resource_group'], self.parameters['network_security_group_name'])
        else:
            network_security_group_name = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkSecurityGroups/%s' % (self.parameters['subscription_id'], self.parameters['resource_group'], self.parameters['network_security_group_name'])
        params['virtualNetworkId']['value'] = network
        params['networkSecurityGroupName']['value'] = network_security_group_name
        params['virtualMachineSize']['value'] = self.parameters['virtual_machine_size']
        params['subnetId']['value'] = subnet
        try:
            resource_client = get_client_from_cli_profile(ResourceManagementClient)
            resource_client.resource_groups.create_or_update(self.parameters['resource_group'], {'location': self.parameters['location']})
            deployment_properties = {'mode': 'Incremental', 'template': template, 'parameters': params}
            resource_client.deployments.begin_create_or_update(self.parameters['resource_group'], self.parameters['name'], Deployment(properties=deployment_properties))
        except CloudError as error:
            self.module.fail_json(msg='Error in deploy_azure: %s' % to_native(error), exception=traceback.format_exc())
        time.sleep(120)
        retries = 30
        while retries > 0:
            occm_resp, error = self.na_helper.check_occm_status(self.rest_api, client_id)
            if error is not None:
                self.module.fail_json(msg='Error: Not able to get occm status: %s, %s' % (str(error), str(occm_resp)))
            if occm_resp['agent']['status'] == 'active':
                break
            else:
                time.sleep(30)
            retries -= 1
        if retries == 0:
            return self.module.fail_json(msg='Taking too long for OCCM agent to be active or not properly setup')
        try:
            compute_client = get_client_from_cli_profile(ComputeManagementClient)
            vm = compute_client.virtual_machines.get(self.parameters['resource_group'], self.parameters['name'])
        except CloudError as error:
            return self.module.fail_json(msg='Error in deploy_azure (get identity): %s' % to_native(error), exception=traceback.format_exc())
        principal_id = vm.identity.principal_id
        return (client_id, principal_id)

    def register_agent_to_service(self):
        """
        Register agent to service and collect userdata by setting up connector
        :return: UserData, ClientID
        """
        if '/subscriptions' in self.parameters['vnet_name']:
            network = self.parameters['vnet_name']
        elif self.parameters.get('vnet_resource_group') is not None:
            network = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s' % (self.parameters['subscription_id'], self.parameters['vnet_resource_group'], self.parameters['vnet_name'])
        else:
            network = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s' % (self.parameters['subscription_id'], self.parameters['resource_group'], self.parameters['vnet_name'])
        if '/subscriptions' in self.parameters['subnet_name']:
            subnet = self.parameters['subnet_name']
        else:
            subnet = '%s/subnets/%s' % (network, self.parameters['subnet_name'])
        if self.parameters.get('account_id') is None:
            response, error = self.na_helper.get_or_create_account(self.rest_api)
            if error is not None:
                self.module.fail_json(msg='Error: unexpected response on getting account: %s, %s' % (str(error), str(response)))
            self.parameters['account_id'] = response
        headers = {'X-User-Token': self.rest_api.token_type + ' ' + self.rest_api.token}
        body = {'accountId': self.parameters['account_id'], 'name': self.parameters['name'], 'company': self.parameters['company'], 'placement': {'provider': 'AZURE', 'region': self.parameters['location'], 'network': network, 'subnet': subnet}, 'extra': {'proxy': {'proxyUrl': self.parameters.get('proxy_url'), 'proxyUserName': self.parameters.get('proxy_user_name'), 'proxyPassword': self.parameters.get('proxy_password')}}}
        register_url = '%s/agents-mgmt/connector-setup' % self.rest_api.environment_data['CLOUD_MANAGER_HOST']
        response, error, dummy = self.rest_api.post(register_url, body, header=headers)
        if error is not None:
            self.module.fail_json(msg='Error: unexpected response on getting userdata for connector setup: %s, %s' % (str(error), str(response)))
        client_id = response['clientId']
        proxy_certificates = []
        if self.parameters.get('proxy_certificates') is not None:
            for each in self.parameters['proxy_certificates']:
                try:
                    data = open(each, 'r').read()
                except OSError:
                    self.module.fail_json(msg='Error: Could not open/read file of proxy_certificates: %s' % str(each))
                encoded_certificate = base64.b64encode(data)
                proxy_certificates.append(encoded_certificate)
        if proxy_certificates:
            response['proxySettings']['proxyCertificates'] = proxy_certificates
        return (response, client_id)

    def delete_azure_occm(self):
        """
        Delete OCCM
        :return:
            None
        """
        try:
            compute_client = get_client_from_cli_profile(ComputeManagementClient)
            vm_delete = compute_client.virtual_machines.begin_delete(self.parameters['resource_group'], self.parameters['name'])
            while not vm_delete.done():
                vm_delete.wait(2)
        except CloudError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        try:
            network_client = get_client_from_cli_profile(NetworkManagementClient)
            interface_delete = network_client.network_interfaces.begin_delete(self.parameters['resource_group'], self.parameters['name'] + '-nic')
            while not interface_delete.done():
                interface_delete.wait(2)
        except CloudError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        try:
            storage_client = get_client_from_cli_profile(StorageManagementClient)
            storage_client.storage_accounts.delete(self.parameters['resource_group'], self.parameters['storage_account'])
        except CloudError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        try:
            network_client = get_client_from_cli_profile(NetworkManagementClient)
            public_ip_addresses_delete = network_client.public_ip_addresses.begin_delete(self.parameters['resource_group'], self.parameters['name'] + '-ip')
            while not public_ip_addresses_delete.done():
                public_ip_addresses_delete.wait(2)
        except CloudError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        try:
            resource_client = get_client_from_cli_profile(ResourceManagementClient)
            deployments_delete = resource_client.deployments.begin_delete(self.parameters['resource_group'], self.parameters['name'] + '-ip')
            while not deployments_delete.done():
                deployments_delete.wait(5)
        except CloudError as error:
            self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
        retries = 16
        while retries > 0:
            occm_resp, error = self.na_helper.check_occm_status(self.rest_api, self.parameters['client_id'])
            if error is not None:
                self.module.fail_json(msg='Error: Not able to get occm status: %s, %s' % (str(error), str(occm_resp)))
            if occm_resp['agent']['status'] != 'active':
                break
            else:
                time.sleep(10)
            retries -= 1
        if retries == 0:
            return self.module.fail_json(msg='Taking too long for instance to finish terminating')
        client = self.rest_api.format_client_id(self.parameters['client_id'])
        error = self.na_helper.delete_occm_agents(self.rest_api, [{'agentId': client}])
        if error:
            self.module.fail_json(msg='Error: unexpected response on deleting OCCM: %s' % str(error))

    def apply(self):
        """
        Apply action to the Cloud Manager connector for AZURE
        :return: None
        """
        client_id = None
        principal_id = None
        if not self.module.check_mode:
            if self.parameters['state'] == 'present':
                client_id, principal_id = self.deploy_azure()
                self.na_helper.changed = True
            elif self.parameters['state'] == 'absent':
                get_deploy = self.get_deploy_azure_vm()
                if get_deploy:
                    self.delete_azure_occm()
                    self.na_helper.changed = True
        self.module.exit_json(changed=self.na_helper.changed, msg={'client_id': client_id, 'principal_id': principal_id})