from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _populate_interfaces_configuration_interface(self, interface_table):
    interfaces = dict()
    for key in interface_table:
        i = interface_table[key]
        interface = dict()
        interface['admin_state'] = i[6].lower()
        interface['mdix'] = i[8].lower()
        interfaces[interface_canonical_name(i[0])] = interface
    return interfaces