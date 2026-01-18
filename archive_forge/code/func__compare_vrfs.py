from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.snmp_server import (
def _compare_vrfs(self, want, have):
    wvrfs = want.get('vrfs', {})
    hvrfs = have.get('vrfs', {})
    for name, entry in iteritems(wvrfs):
        begin = len(self.commands)
        vrf_have = hvrfs.pop(name, {})
        self._compare_lists(want=entry, have=vrf_have)
        if len(self.commands) != begin:
            self._remove_snmp_server(begin)
            self.commands.insert(begin, self._tmplt.render({'vrf': entry.get('vrf')}, 'vrfs', False))
    for name, entry in iteritems(hvrfs):
        self.addcmd(entry, 'vrfs', True)