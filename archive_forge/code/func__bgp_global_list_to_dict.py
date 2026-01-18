from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.bgp_global import (
def _bgp_global_list_to_dict(self, entry):
    for name, proc in iteritems(entry):
        if 'neighbor' in proc:
            neigh_dict = {}
            for entry in proc.get('neighbor', []):
                neigh_dict.update({entry['address']: entry})
            proc['neighbor'] = neigh_dict
        if 'network' in proc:
            network_dict = {}
            for entry in proc.get('network', []):
                network_dict.update({entry['address']: entry})
            proc['network'] = network_dict
        if 'aggregate_address' in proc:
            agg_dict = {}
            for entry in proc.get('aggregate_address', []):
                agg_dict.update({entry['prefix']: entry})
            proc['aggregate_address'] = agg_dict
        if 'redistribute' in proc:
            redis_dict = {}
            for entry in proc.get('redistribute', []):
                redis_dict.update({entry['protocol']: entry})
            proc['redistribute'] = redis_dict