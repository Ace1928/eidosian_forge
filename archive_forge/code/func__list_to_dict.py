from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv2 import (
def _list_to_dict(self, param):
    for _pid, proc in param.items():
        for area in proc.get('areas', []):
            area['ranges'] = {entry['address']: entry for entry in area.get('ranges', [])}
            area['filter_list'] = {entry['direction']: entry for entry in area.get('filter_list', [])}
        proc['areas'] = {entry['area_id']: entry for entry in proc.get('areas', [])}
        distribute_list = proc.get('distribute_list', {})
        if 'acls' in distribute_list:
            distribute_list['acls'] = {entry['name']: entry for entry in distribute_list['acls']}
        passive_interfaces = proc.get('passive_interfaces', {}).get('interface', {})
        if passive_interfaces.get('name'):
            passive_interfaces['name'] = {entry: entry for entry in passive_interfaces['name']}
        if proc.get('network'):
            proc['network'] = {entry['address']: entry for entry in proc['network']}