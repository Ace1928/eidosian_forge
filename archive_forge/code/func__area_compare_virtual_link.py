from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.ospfv2 import (
def _area_compare_virtual_link(self, want, have):
    parsers = ['virtual_link.authentication', 'virtual_link.authentication_key', 'virtual_link.authentication.message_digest', 'virtual_link.hello_interval', 'virtual_link.dead_interval', 'virtual_link.retransmit_interval']
    self.compare(parsers=parsers, want=want, have=have)