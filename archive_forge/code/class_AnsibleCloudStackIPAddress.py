from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackIPAddress(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackIPAddress, self).__init__(module)
        self.returns = {'ipaddress': 'ip_address'}

    def get_ip_address(self, key=None):
        if self.ip_address:
            return self._get_by_key(key, self.ip_address)
        args = {'ipaddress': self.module.params.get('ip_address'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'vpcid': self.get_vpc(key='id')}
        ip_addresses = self.query_api('listPublicIpAddresses', **args)
        if ip_addresses:
            tags = self.module.params.get('tags')
            for ip_addr in ip_addresses['publicipaddress']:
                if ip_addr['ipaddress'] == args['ipaddress'] != '':
                    self.ip_address = ip_addresses['publicipaddress'][0]
                elif tags:
                    if sorted([tag for tag in tags if tag in ip_addr['tags']]) == sorted(tags):
                        self.ip_address = ip_addr
        return self._get_by_key(key, self.ip_address)

    def present_ip_address(self):
        ip_address = self.get_ip_address()
        if not ip_address:
            ip_address = self.associate_ip_address(ip_address)
        if ip_address:
            ip_address = self.ensure_tags(resource=ip_address, resource_type='publicipaddress')
        return ip_address

    def associate_ip_address(self, ip_address):
        self.result['changed'] = True
        args = {'ipaddress': self.module.params.get('ip_address'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'networkid': self.get_network(key='id') if not self.module.params.get('vpc') else None, 'zoneid': self.get_zone(key='id'), 'vpcid': self.get_vpc(key='id')}
        ip_address = None
        if not self.module.check_mode:
            res = self.query_api('associateIpAddress', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                ip_address = self.poll_job(res, 'ipaddress')
        return ip_address

    def disassociate_ip_address(self):
        ip_address = self.get_ip_address()
        if not ip_address:
            return None
        if ip_address['isstaticnat']:
            self.module.fail_json(msg='IP address is allocated via static nat')
        self.result['changed'] = True
        if not self.module.check_mode:
            self.module.params['tags'] = []
            ip_address = self.ensure_tags(resource=ip_address, resource_type='publicipaddress')
            res = self.query_api('disassociateIpAddress', id=ip_address['id'])
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'ipaddress')
        return ip_address