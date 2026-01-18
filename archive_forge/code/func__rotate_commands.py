from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.ospfv3 import (
def _rotate_commands(self, begin=0):
    for cmd in self.commands[begin:]:
        negate = re.match('^no .*', cmd)
        if negate:
            self.commands.insert(begin, self.commands.pop(self.commands.index(cmd)))
            begin += 1