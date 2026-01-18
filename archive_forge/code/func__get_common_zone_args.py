from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _get_common_zone_args(self):
    args = {'name': self.module.params.get('name'), 'dns1': self.module.params.get('dns1'), 'dns2': self.module.params.get('dns2'), 'internaldns1': self.get_or_fallback('internal_dns1', 'dns1'), 'internaldns2': self.get_or_fallback('internal_dns2', 'dns2'), 'ipv6dns1': self.module.params.get('dns1_ipv6'), 'ipv6dns2': self.module.params.get('dns2_ipv6'), 'networktype': self.module.params.get('network_type'), 'domain': self.module.params.get('network_domain'), 'localstorageenabled': self.module.params.get('local_storage_enabled'), 'guestcidraddress': self.module.params.get('guest_cidr_address'), 'dhcpprovider': self.module.params.get('dhcp_provider')}
    state = self.module.params.get('state')
    if state in ['enabled', 'disabled']:
        args['allocationstate'] = state.capitalize()
    return args