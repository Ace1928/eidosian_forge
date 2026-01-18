from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.logging_global import (
def _logging_global_list_to_dict(self, entry):
    if 'hosts' in entry:
        hosts_dict = {}
        for el in entry['hosts']:
            hosts_dict.update({el['name']: el})
        entry['hosts'] = hosts_dict
    if 'vrfs' in entry:
        vrf_dict = {}
        for el in entry['vrfs']:
            vrf_dict.update({el['name']: el})
        entry['vrfs'] = vrf_dict
        for k, v in iteritems(entry['vrfs']):
            self._logging_global_list_to_dict(v)