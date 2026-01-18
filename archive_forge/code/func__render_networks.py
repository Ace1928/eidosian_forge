from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.cli.config.bgp.address_family import (
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.cli.config.bgp.neighbors import (
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_networks(self, config=None):
    commands = list()
    safe_list = list()
    for entry in self.get_value('config.networks'):
        network = entry['prefix']
        if entry['masklen']:
            network = '%s/%s' % (entry['prefix'], entry['masklen'])
        safe_list.append(network)
        cmd = 'network %s' % network
        if entry['route_map']:
            cmd += ' route-map %s' % entry['route_map']
        if not config or cmd not in config:
            commands.append(cmd)
    if self.params['operation'] == 'replace':
        if config:
            matches = re.findall('network (\\S+)', config, re.M)
            for entry in set(matches).difference(safe_list):
                commands.append('no network %s' % entry)
    return commands