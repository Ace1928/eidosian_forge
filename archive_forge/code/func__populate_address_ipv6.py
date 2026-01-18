from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _populate_address_ipv6(self, ip_table):
    ips = list()
    for key in ip_table:
        ip = ip_table[key][3]
        interface = interface_canonical_name(ip_table[key][0])
        ips.append(ip)
        self._new_interface(interface)
        if 'ipv6' not in self.facts['interfaces'][interface]:
            self.facts['interfaces'][interface]['ipv6'] = list()
        self.facts['interfaces'][interface]['ipv6'].append(dict(address=ip))
    return ips