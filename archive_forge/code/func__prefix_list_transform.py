from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.prefix_lists import (
def _prefix_list_transform(self, entry):
    for afi, value in iteritems(entry):
        if 'prefix_lists' in value:
            for plist in value['prefix_lists']:
                plist.update({'afi': afi})
                if 'entries' in plist:
                    for seq in plist['entries']:
                        seq.update({'afi': afi, 'name': plist['name']})
                    plist['entries'] = {x['sequence']: x for x in plist['entries']}
            value['prefix_lists'] = {entry['name']: entry for entry in value['prefix_lists']}