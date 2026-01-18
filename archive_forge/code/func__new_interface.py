from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def _new_interface(self, interface):
    if interface in self.facts['interfaces']:
        return
    else:
        self.facts['interfaces'][interface] = dict()
        self.facts['interfaces'][interface]['mtu'] = self._mtu
        self.facts['interfaces'][interface]['admin_state'] = 'up'
        self.facts['interfaces'][interface]['description'] = None
        self.facts['interfaces'][interface]['state'] = 'up'
        self.facts['interfaces'][interface]['bandwith'] = None
        self.facts['interfaces'][interface]['duplex'] = None
        self.facts['interfaces'][interface]['negotiation'] = None
        self.facts['interfaces'][interface]['control'] = None
        return