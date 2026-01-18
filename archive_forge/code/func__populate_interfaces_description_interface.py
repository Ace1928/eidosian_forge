from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _populate_interfaces_description_interface(self, interface_table):
    interfaces = dict()
    for key in interface_table:
        i = interface_table[key]
        interface = dict()
        interface['description'] = i[1]
        if interface['description'] == '':
            interface['description'] = None
        interfaces[interface_canonical_name(i[0])] = interface
    return interfaces