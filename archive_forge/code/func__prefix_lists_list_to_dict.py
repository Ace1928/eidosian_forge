from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.prefix_lists import (
def _prefix_lists_list_to_dict(self, entry):
    for afi, plist in iteritems(entry):
        if 'prefix_lists' in plist:
            pl_dict = {}
            for el in plist['prefix_lists']:
                if 'entries' in el:
                    ent_dict = {}
                    for en in el['entries']:
                        if 'sequence' not in en.keys():
                            num = 'seq'
                        else:
                            num = en['sequence']
                        ent_dict.update({num: en})
                    el['entries'] = ent_dict
            for el in plist['prefix_lists']:
                pl_dict.update({el['name']: el})
            plist['prefix_lists'] = pl_dict