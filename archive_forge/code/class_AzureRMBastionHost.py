from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMBastionHost(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), resource_group=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), ip_configurations=dict(type='list', elements='dict', options=dict(name=dict(type='str'), subnet=dict(type='dict', options=subnet_spec), public_ip_address=dict(type='dict', options=public_ip_address_spec), private_ip_allocation_method=dict(type='str', choices=['Static', 'Dynamic']))), sku=dict(type='dict', options=sku_spec), enable_tunneling=dict(type='bool'), enable_shareable_link=dict(type='bool'), enable_ip_connect=dict(type='bool'), enable_file_copy=dict(type='bool'), scale_units=dict(type='int'), disable_copy_paste=dict(type='bool'))
        self.name = None
        self.resource_group = None
        self.location = None
        self.tags = None
        self.state = None
        self.results = dict(changed=False)
        self.body = {}
        super(AzureRMBastionHost, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True, facts_module=False)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        old_response = self.get_item()
        result = None
        changed = False
        if not self.location:
            resource_group = self.get_resource_group(self.resource_group)
            self.location = resource_group.location
        self.body['location'] = self.location
        self.body['tags'] = self.tags
        if self.state == 'present':
            if old_response:
                update_tags, tags = self.update_tags(old_response['tags'])
                if update_tags:
                    changed = True
                self.body['tags'] = tags
                if self.body.get('disable_copy_paste') is not None:
                    if bool(self.body.get('disable_copy_paste')) != bool(old_response['disable_copy_paste']):
                        changed = True
                else:
                    self.body['disable_copy_paste'] = old_response['disable_copy_paste']
                if self.body.get('enable_file_copy') is not None:
                    if bool(self.body.get('enable_file_copy')) != bool(old_response['enable_file_copy']):
                        changed = True
                else:
                    self.body['enable_file_copy'] = old_response['enable_file_copy']
                if self.body.get('enable_ip_connect') is not None:
                    if bool(self.body.get('enable_ip_connect')) != bool(old_response['enable_ip_connect']):
                        changed = True
                else:
                    self.body['enable_ip_connect'] = old_response['enable_ip_connect']
                if self.body.get('enable_shareable_link') is not None:
                    if bool(self.body.get('enable_shareable_link')) != bool(old_response['enable_shareable_link']):
                        changed = True
                else:
                    self.body['enable_shareable_link'] = old_response['enable_shareable_link']
                if self.body.get('enable_tunneling') is not None:
                    if bool(self.body.get('enable_tunneling')) != bool(old_response['enable_tunneling']):
                        changed = True
                else:
                    self.body['enable_tunneling'] = old_response['enable_tunneling']
                if self.body.get('scale_units') is not None:
                    if self.body.get('scale_units') != old_response['scale_units']:
                        changed = True
                else:
                    self.body['scale_units'] = old_response['scale_units']
                if self.body.get('sku') is not None:
                    if self.body.get('sku') != old_response['sku']:
                        changed = True
                else:
                    self.body['sku'] = old_response['sku']
                if self.body.get('ip_configurations') is not None:
                    if self.body['ip_configurations'] != old_response['ip_configurations']:
                        self.fail('Bastion Host IP configuration not support to update!')
                else:
                    self.body['ip_configurations'] = old_response['ip_configurations']
            else:
                changed = True
            if changed:
                if self.check_mode:
                    self.log('Check mode test. The bastion host is exist, will be create or updated')
                else:
                    result = self.create_or_update(self.body)
            elif self.check_mode:
                self.log('Check mode test. The Azure Bastion Host is exist, No operation in this task')
            else:
                self.log('The Azure Bastion Host is exist, No operation in this task')
                result = old_response
        elif old_response:
            changed = True
            if self.check_mode:
                self.log('Check mode test. The bastion host is exist, will be deleted')
            else:
                result = self.delete_resource()
        elif self.check_mode:
            self.log("The bastion host isn't exist, no action")
        else:
            self.log("The bastion host isn't exist, don't need to delete")
        self.results['bastion_host'] = result
        self.results['changed'] = changed
        return self.results

    def get_item(self):
        self.log('Get properties for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.bastion_hosts.get(self.resource_group, self.name)
            return self.bastion_to_dict(response)
        except ResourceNotFoundError:
            self.log('Could not get info for {0} in {1}'.format(self.name, self.resource_group))
        return []

    def create_or_update(self, parameters):
        self.log('Create or update the bastion host for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.bastion_hosts.begin_create_or_update(self.resource_group, self.name, parameters)
            result = self.network_client.bastion_hosts.get(self.resource_group, self.name)
            return self.bastion_to_dict(result)
        except Exception as ec:
            self.fail('Create or Update {0} in {1} failed, mesage {2}'.format(self.name, self.resource_group, ec))
        return []

    def delete_resource(self):
        self.log('delete the bastion host for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.bastion_hosts.begin_delete(self.resource_group, self.name)
        except Exception as ec:
            self.fail('Delete {0} in {1} failed, message {2}'.format(self.name, self.resource_group, ec))
        return []

    def bastion_to_dict(self, bastion_info):
        bastion = bastion_info.as_dict()
        result = dict(id=bastion.get('id'), name=bastion.get('name'), type=bastion.get('type'), etag=bastion.get('etag'), location=bastion.get('location'), tags=bastion.get('tags'), sku=dict(), ip_configurations=list(), dns_name=bastion.get('dns_name'), provisioning_state=bastion.get('provisioning_state'), scale_units=bastion.get('scale_units'), disable_copy_paste=bastion.get('disable_copy_paste'), enable_file_copy=bastion.get('enable_file_copy'), enable_ip_connect=bastion.get('enable_ip_connect'), enable_shareable_link=bastion.get('enable_tunneling'), enable_tunneling=bastion.get('enable_tunneling'))
        if bastion.get('sku'):
            result['sku']['name'] = bastion['sku']['name']
        if bastion.get('ip_configurations'):
            for items in bastion['ip_configurations']:
                result['ip_configurations'].append({'name': items['name'], 'subnet': dict(id=items['subnet']['id']), 'public_ip_address': dict(id=items['public_ip_address']['id']), 'private_ip_allocation_method': items['private_ip_allocation_method']})
        return result