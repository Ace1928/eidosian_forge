from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.snmp_server import (
def _move_negate_commands(self):
    command_set = []
    for cmd in self.commands:
        if re.search('delete service snmp', cmd):
            command_set.insert(0, cmd)
        else:
            command_set.append(cmd)
    self.commands = command_set