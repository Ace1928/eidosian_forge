from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_global import (
def _vrfs_compare(self, want, have):
    """Custom handling of VRFs option
        :params want: the want BGP dictionary
        :params have: the have BGP dictionary
        """
    wvrfs = want.get('vrfs', {})
    hvrfs = have.get('vrfs', {})
    for name, entry in iteritems(wvrfs):
        begin = len(self.commands)
        vrf_have = hvrfs.pop(name, {})
        self._compare_rpki_server(want=entry, have=vrf_have)
        self._compare_neighbors(want=entry, have=vrf_have)
        self.compare(parsers=self.parsers, want=entry, have=vrf_have)
        if len(self.commands) != begin:
            self.commands.insert(begin, self._tmplt.render({'vrf': entry.get('vrf')}, 'vrf', False))
    for name, entry in iteritems(hvrfs):
        if self._check_af('vrf', name):
            self._module.fail_json(msg='VRF {0} has address-family configurations. Please use the iosxr_bgp_address_family module to remove those first.'.format(name))
        else:
            self.addcmd(entry, 'vrf', True)