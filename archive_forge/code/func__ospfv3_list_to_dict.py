from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospfv3 import (
def _ospfv3_list_to_dict(self, entry):
    for _pid, proc in iteritems(entry):
        proc['areas'] = {entry['area_id']: entry for entry in proc.get('areas', [])}
        af = proc.get('address_family')
        if af:
            for area in af.get('areas', []):
                area['ranges'] = {entry['prefix']: entry for entry in area.get('ranges', [])}
                area['filter_list'] = {entry['direction']: entry for entry in area.get('filter_list', [])}
            af['areas'] = {entry['area_id']: entry for entry in af.get('areas', [])}
            af['summary_address'] = {entry['prefix']: entry for entry in af.get('summary_address', [])}
            af['redistribute'] = {(entry.get('id'), entry['protocol']): entry for entry in af.get('redistribute', [])}
        if 'vrfs' in proc:
            proc['vrfs'] = {entry['vrf']: entry for entry in proc.get('vrfs', [])}
            self._ospfv3_list_to_dict(proc['vrfs'])