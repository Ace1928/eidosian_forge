from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospf_interfaces import (
def _compare_processes(self, afi, want, have):
    for w_id, wproc in want.items():
        hproc = have.pop(w_id, {})
        hproc['afi'] = wproc['afi'] = afi
        self.compare(['area'], wproc, hproc)
        marea_dict = {'afi': afi, 'process_id': wproc['process_id']}
        for area in wproc.get('multi_areas', []):
            if area not in hproc.get('multi_areas', []):
                marea_dict['area'] = area
                self.addcmd(marea_dict, 'processes_multi_areas', negate=False)
        for area in hproc.get('multi_areas', []):
            if area not in wproc.get('multi_areas', []):
                marea_dict['area'] = area
                self.addcmd(marea_dict, 'processes_multi_areas', negate=True)
    for hproc in have.values():
        hproc['afi'] = afi
        self.addcmd(hproc, 'area', negate=True)
        marea_dict = {'afi': afi, 'process_id': hproc['process_id']}
        for area in hproc.get('multi_areas', []):
            marea_dict['area'] = area
            self.addcmd(marea_dict, 'processes_multi_areas', negate=True)