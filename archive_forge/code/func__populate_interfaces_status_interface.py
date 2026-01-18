from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _populate_interfaces_status_interface(self, interface_table):
    interfaces = dict()
    for key in interface_table:
        i = interface_table[key]
        interface = dict()
        interface['state'] = i[6].lower()
        interface['type'] = i[1]
        interface['mtu'] = self._mtu
        interface['duplex'] = i[2].lower()
        interface['negotiation'] = i[4].lower()
        interface['control'] = i[5].lower()
        interface['presure'] = i[7].lower()
        interface['mode'] = i[8].lower()
        if i[6] == 'Up':
            interface['bandwith'] = int(i[3]) * 1000
        else:
            interface['bandwith'] = None
        for key in interface:
            if interface[key] == '--':
                interface[key] = None
        interfaces[interface_canonical_name(i[0])] = interface
    return interfaces