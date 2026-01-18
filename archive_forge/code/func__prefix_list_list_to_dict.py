from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.prefix_lists import (
def _prefix_list_list_to_dict(self, entry):
    for afi, value in iteritems(entry):
        if 'prefix_lists' in value:
            for pl in value['prefix_lists']:
                pl.update({'afi': afi})
                if 'entries' in pl:
                    for entry in pl['entries']:
                        entry.update({'afi': afi, 'name': pl['name']})
                    pl['entries'] = {x['sequence']: x for x in pl['entries']}
            value['prefix_lists'] = {entry['name']: entry for entry in value['prefix_lists']}