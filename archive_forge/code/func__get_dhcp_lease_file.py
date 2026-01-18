from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.facts import ansible_collector, default_collectors
def _get_dhcp_lease_file(self):
    """Return the path of the lease file."""
    default_iface = self.facts['default_ipv4']['interface']
    dhcp_lease_file_locations = ['/var/lib/dhcp/dhclient.%s.leases' % default_iface, '/var/lib/dhclient/dhclient-%s.leases' % default_iface, '/var/lib/dhclient/dhclient--%s.lease' % default_iface, '/var/db/dhclient.leases.%s' % default_iface]
    for file_path in dhcp_lease_file_locations:
        if os.path.exists(file_path):
            return file_path
    module.fail_json(msg='Could not find dhclient leases file.')