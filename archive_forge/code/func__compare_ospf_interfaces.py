from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospf_interfaces import (
def _compare_ospf_interfaces(self, want, have):
    waf = want.get('address_family', {})
    haf = have.get('address_family', {})
    for afi in ('ipv4', 'ipv6'):
        witem = waf.pop(afi, {})
        hitem = haf.pop(afi, {})
        self.compare(['authentication.key_chain'], want=witem, have=hitem)
        witem.get('authentication', {}).pop('key_chain', None)
        hitem.get('authentication', {}).pop('key_chain', None)
        if witem.get('passive_interface') is False and 'passive_interface' not in hitem:
            hitem['passive_interface'] = True
        if 'passive_interface' in hitem and witem.get('default_passive_interface'):
            self.commands.append(self._generate_passive_intf(witem))
        self.compare(parsers=self.parsers, want=witem, have=hitem)
        for area in witem.get('multi_areas', []):
            if area not in hitem.get('multi_areas', []):
                self.addcmd({'afi': afi, 'area': area}, 'multi_areas', negate=False)
        for area in hitem.get('multi_areas', []):
            if area not in witem.get('multi_areas', []):
                self.addcmd({'afi': afi, 'area': area}, 'multi_areas', negate=True)
        self._compare_processes(afi, witem.get('processes', {}), hitem.get('processes', {}))