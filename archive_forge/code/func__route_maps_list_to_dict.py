from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.route_maps import (
def _route_maps_list_to_dict(self, entry):
    entry = {x['route_map']: x for x in entry}
    for rmap, data in iteritems(entry):
        if 'entries' in data:
            for x in data['entries']:
                x.update({'route_map': rmap})
            data['entries'] = {(rmap, entry.get('sequence')): entry for entry in data['entries']}
    return entry