from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _populate_address_ipv4(self, ip_table):
    ips = list()
    for key in ip_table:
        cidr = ip_table[key][0]
        interface = interface_canonical_name(ip_table[key][1])
        ip, mask = cidr.split('/')
        ips.append(ip)
        self._new_interface(interface)
        if 'ipv4' not in self.facts['interfaces'][interface]:
            self.facts['interfaces'][interface]['ipv4'] = list()
        self.facts['interfaces'][interface]['ipv4'].append(dict(address=ip, subnet=mask))
    return ips