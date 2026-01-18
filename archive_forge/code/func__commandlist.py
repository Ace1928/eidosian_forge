from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ntp_global import (
def _commandlist(self, haved):
    commandlist = []
    for k, have in iteritems(haved):
        for ck, cval in iteritems(have):
            if ck != 'options' and ck not in commandlist:
                commandlist.append(ck)
    return commandlist