from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
class AzureRMPublicIPAddress(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), version=dict(type='str', default='ipv4', choices=['ipv4', 'ipv6']), allocation_method=dict(type='str', default='dynamic', choices=['Dynamic', 'Static', 'dynamic', 'static']), domain_name=dict(type='str', aliases=['domain_name_label']), sku=dict(type='str', choices=['Basic', 'Standard', 'basic', 'standard']), ip_tags=dict(type='list', elements='dict', options=ip_tag_spec), idle_timeout=dict(type='int'), zones=dict(type='list', elements='str', choices=['1', '2', '3']))
        self.resource_group = None
        self.name = None
        self.location = None
        self.state = None
        self.tags = None
        self.zones = None
        self.allocation_method = None
        self.domain_name = None
        self.sku = None
        self.version = None
        self.ip_tags = None
        self.idle_timeout = None
        self.results = dict(changed=False, state=dict())
        super(AzureRMPublicIPAddress, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        results = dict()
        changed = False
        pip = None
        self.allocation_method = self.allocation_method.capitalize() if self.allocation_method else None
        self.sku = self.sku.capitalize() if self.sku else None
        self.version = 'IPv4' if self.version == 'ipv4' else 'IPv6'
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        try:
            self.log('Fetch public ip {0}'.format(self.name))
            pip = self.network_client.public_ip_addresses.get(self.resource_group, self.name)
            self.check_provisioning_state(pip, self.state)
            self.log('PIP {0} exists'.format(self.name))
            if self.state == 'present':
                results = pip_to_dict(pip)
                domain_lable = results['dns_settings'].get('domain_name_label')
                if self.domain_name is not None and ((self.domain_name or domain_lable) and self.domain_name != domain_lable):
                    self.log('CHANGED: domain_name_label')
                    changed = True
                    results['dns_settings']['domain_name_label'] = self.domain_name
                if self.allocation_method.lower() != results['public_ip_allocation_method'].lower():
                    self.log('CHANGED: allocation_method')
                    changed = True
                    results['public_ip_allocation_method'] = self.allocation_method
                if self.sku and self.sku != results['sku']:
                    self.log('CHANGED: sku')
                    changed = True
                    results['sku'] = self.sku
                if self.version.lower() != results['public_ip_address_version'].lower():
                    self.log('CHANGED: version')
                    changed = True
                    results['public_ip_address_version'] = self.version
                if self.idle_timeout and self.idle_timeout != results['idle_timeout_in_minutes']:
                    self.log('CHANGED: idle_timeout')
                    changed = True
                    results['idle_timeout_in_minutes'] = self.idle_timeout
                if self.zones and self.zones != results['zones']:
                    self.log('Zones defined do not same with existing zones')
                    changed = False
                    self.fail('ResourceAvailabilityZonesCannotBeModified: defines is {0}, existing is {1}'.format(self.zones, results['zones']))
                if str(self.ip_tags or []) != str(results.get('ip_tags') or []):
                    self.log('CHANGED: ip_tags')
                    changed = True
                    results['ip_tags'] = self.ip_tags
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
            elif self.state == 'absent':
                self.log("CHANGED: public ip {0} exists but requested state is 'absent'".format(self.name))
                changed = True
        except ResourceNotFoundError:
            self.log('Public ip {0} does not exist'.format(self.name))
            if self.state == 'present':
                self.log("CHANGED: pip {0} does not exist but requested state is 'present'".format(self.name))
                changed = True
        self.results['state'] = results
        self.results['changed'] = changed
        if self.check_mode:
            return results
        if changed:
            if self.state == 'present':
                if not pip:
                    self.log('Create new Public IP {0}'.format(self.name))
                    pip = self.network_models.PublicIPAddress(location=self.location, public_ip_address_version=self.version, public_ip_allocation_method=self.allocation_method, sku=self.network_models.PublicIPAddressSku(name=self.sku) if self.sku else None, idle_timeout_in_minutes=self.idle_timeout if self.idle_timeout and self.idle_timeout > 0 else None, zones=self.zones)
                    if self.ip_tags:
                        pip.ip_tags = [self.network_models.IpTag(ip_tag_type=x['type'], tag=x['value']) for x in self.ip_tags]
                    if self.tags:
                        pip.tags = self.tags
                    if self.domain_name:
                        pip.dns_settings = self.network_models.PublicIPAddressDnsSettings(domain_name_label=self.domain_name)
                else:
                    self.log('Update Public IP {0}'.format(self.name))
                    pip = self.network_models.PublicIPAddress(location=results['location'], public_ip_allocation_method=results['public_ip_allocation_method'], sku=self.network_models.PublicIPAddressSku(name=self.sku) if self.sku else None, tags=results['tags'], zones=results['zones'])
                    if self.domain_name:
                        pip.dns_settings = self.network_models.PublicIPAddressDnsSettings(domain_name_label=self.domain_name)
                self.results['state'] = self.create_or_update_pip(pip)
            elif self.state == 'absent':
                self.log('Delete public ip {0}'.format(self.name))
                self.delete_pip()
        return self.results

    def create_or_update_pip(self, pip):
        try:
            poller = self.network_client.public_ip_addresses.begin_create_or_update(self.resource_group, self.name, pip)
            pip = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error creating or updating {0} - {1}'.format(self.name, str(exc)))
        return pip_to_dict(pip)

    def delete_pip(self):
        try:
            poller = self.network_client.public_ip_addresses.begin_delete(self.resource_group, self.name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting {0} - {1}'.format(self.name, str(exc)))
        self.results['state']['status'] = 'Deleted'
        return True