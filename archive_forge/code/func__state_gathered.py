from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def _state_gathered(self, have):
    """The command generator when state is gathered

        :rtype: A list
        :returns: the commands necessary to reproduce the current configuration
        """
    commands = []
    want = {}
    commands.append(self.set_commands(want, have))
    return commands