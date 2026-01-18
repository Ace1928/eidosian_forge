from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackZone(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackZone, self).__init__(module)
        self.returns = {'dns1': 'dns1', 'dns2': 'dns2', 'internaldns1': 'internal_dns1', 'internaldns2': 'internal_dns2', 'ipv6dns1': 'dns1_ipv6', 'ipv6dns2': 'dns2_ipv6', 'domain': 'network_domain', 'networktype': 'network_type', 'securitygroupsenabled': 'securitygroups_enabled', 'localstorageenabled': 'local_storage_enabled', 'guestcidraddress': 'guest_cidr_address', 'dhcpprovider': 'dhcp_provider', 'allocationstate': 'allocation_state', 'zonetoken': 'zone_token'}
        self.zone = None

    def _get_common_zone_args(self):
        args = {'name': self.module.params.get('name'), 'dns1': self.module.params.get('dns1'), 'dns2': self.module.params.get('dns2'), 'internaldns1': self.get_or_fallback('internal_dns1', 'dns1'), 'internaldns2': self.get_or_fallback('internal_dns2', 'dns2'), 'ipv6dns1': self.module.params.get('dns1_ipv6'), 'ipv6dns2': self.module.params.get('dns2_ipv6'), 'networktype': self.module.params.get('network_type'), 'domain': self.module.params.get('network_domain'), 'localstorageenabled': self.module.params.get('local_storage_enabled'), 'guestcidraddress': self.module.params.get('guest_cidr_address'), 'dhcpprovider': self.module.params.get('dhcp_provider')}
        state = self.module.params.get('state')
        if state in ['enabled', 'disabled']:
            args['allocationstate'] = state.capitalize()
        return args

    def get_zone(self):
        if not self.zone:
            args = {}
            uuid = self.module.params.get('id')
            if uuid:
                args['id'] = uuid
                zones = self.query_api('listZones', **args)
                if zones:
                    self.zone = zones['zone'][0]
                    return self.zone
            args['name'] = self.module.params.get('name')
            zones = self.query_api('listZones', **args)
            if zones:
                self.zone = zones['zone'][0]
        return self.zone

    def present_zone(self):
        zone = self.get_zone()
        if zone:
            zone = self._update_zone()
        else:
            zone = self._create_zone()
        return zone

    def _create_zone(self):
        required_params = ['dns1']
        self.module.fail_on_missing_params(required_params=required_params)
        self.result['changed'] = True
        args = self._get_common_zone_args()
        args['domainid'] = self.get_domain(key='id')
        args['securitygroupenabled'] = self.module.params.get('securitygroups_enabled')
        zone = None
        if not self.module.check_mode:
            res = self.query_api('createZone', **args)
            zone = res['zone']
        return zone

    def _update_zone(self):
        zone = self.get_zone()
        args = self._get_common_zone_args()
        args['id'] = zone['id']
        if self.has_changed(args, zone):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateZone', **args)
                zone = res['zone']
        return zone

    def absent_zone(self):
        zone = self.get_zone()
        if zone:
            self.result['changed'] = True
            args = {'id': zone['id']}
            if not self.module.check_mode:
                self.query_api('deleteZone', **args)
        return zone