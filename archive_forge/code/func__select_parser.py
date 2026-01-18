from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.route_maps import (
def _select_parser(self, w):
    parser = ''
    if 'statement' in w.keys() and 'action' in w.keys() and ('sequence' in w.keys()):
        parser = 'route_map.statement.entries'
    elif 'statement' in w.keys() and 'action' in w.keys():
        parser = 'route_map.statement.action'
    elif 'statement' in w.keys():
        parser = 'route_map.statement.name'
    elif 'action' in w.keys() and 'sequence' in w.keys():
        parser = 'route_map.entries'
    elif 'action' in w.keys():
        parser = 'route_map.action'
    else:
        parser = 'route_map.name'
    return parser