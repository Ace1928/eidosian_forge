from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackZoneInfo(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackZoneInfo, self).__init__(module)
        self.returns = {'dns1': 'dns1', 'dns2': 'dns2', 'internaldns1': 'internal_dns1', 'internaldns2': 'internal_dns2', 'ipv6dns1': 'dns1_ipv6', 'ipv6dns2': 'dns2_ipv6', 'domain': 'network_domain', 'networktype': 'network_type', 'securitygroupsenabled': 'securitygroups_enabled', 'localstorageenabled': 'local_storage_enabled', 'guestcidraddress': 'guest_cidr_address', 'dhcpprovider': 'dhcp_provider', 'allocationstate': 'allocation_state', 'zonetoken': 'zone_token'}

    def get_zone(self):
        if self.module.params['zone']:
            zones = [super(AnsibleCloudStackZoneInfo, self).get_zone()]
        else:
            zones = self.query_api('listZones')
            if zones:
                zones = zones['zone']
            else:
                zones = []
        return {'zones': [self.update_result(resource) for resource in zones]}