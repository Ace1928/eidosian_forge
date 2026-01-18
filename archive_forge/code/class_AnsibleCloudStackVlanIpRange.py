from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackVlanIpRange(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVlanIpRange, self).__init__(module)
        self.returns = {'startip': 'start_ip', 'endip': 'end_ip', 'physicalnetworkid': 'physical_network', 'vlan': 'vlan', 'forsystemvms': 'for_systemvms', 'forvirtualnetwork': 'for_virtual_network', 'gateway': 'gateway', 'netmask': 'netmask', 'ip6gateway': 'gateway_ipv6', 'ip6cidr': 'cidr_ipv6', 'startipv6': 'start_ipv6', 'endipv6': 'end_ipv6'}
        self.ip_range = None

    def get_vlan_ip_range(self):
        if not self.ip_range:
            args = {'zoneid': self.get_zone(key='id'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'networkid': self.get_network(key='id')}
            res = self.query_api('listVlanIpRanges', **args)
            if res:
                ip_range_list = res['vlaniprange']
                params = {'startip': self.module.params.get('start_ip'), 'endip': self.get_or_fallback('end_ip', 'start_ip')}
                for ipr in ip_range_list:
                    if params['startip'] == ipr['startip'] and params['endip'] == ipr['endip']:
                        self.ip_range = ipr
                        break
        return self.ip_range

    def present_vlan_ip_range(self):
        ip_range = self.get_vlan_ip_range()
        if not ip_range:
            ip_range = self.create_vlan_ip_range()
        return ip_range

    def create_vlan_ip_range(self):
        self.result['changed'] = True
        vlan = self.module.params.get('vlan')
        args = {'zoneid': self.get_zone(key='id'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'startip': self.module.params.get('start_ip'), 'endip': self.get_or_fallback('end_ip', 'start_ip'), 'netmask': self.module.params.get('netmask'), 'gateway': self.module.params.get('gateway'), 'startipv6': self.module.params.get('start_ipv6'), 'endipv6': self.get_or_fallback('end_ipv6', 'start_ipv6'), 'ip6gateway': self.module.params.get('gateway_ipv6'), 'ip6cidr': self.module.params.get('cidr_ipv6'), 'vlan': self.get_network(key='vlan') if not vlan else vlan, 'networkid': self.get_network(key='id'), 'forvirtualnetwork': self.module.params.get('for_virtual_network'), 'forsystemvms': self.module.params.get('for_system_vms'), 'podid': self.get_pod(key='id')}
        if self.module.params.get('physical_network'):
            args['physicalnetworkid'] = self.get_physical_network(key='id')
        if not self.module.check_mode:
            res = self.query_api('createVlanIpRange', **args)
            self.ip_range = res['vlan']
        return self.ip_range

    def absent_vlan_ip_range(self):
        ip_range = self.get_vlan_ip_range()
        if ip_range:
            self.result['changed'] = True
            args = {'id': ip_range['id']}
            if not self.module.check_mode:
                self.query_api('deleteVlanIpRange', **args)
        return ip_range