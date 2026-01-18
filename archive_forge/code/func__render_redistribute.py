from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.cli.config.bgp.neighbors import (
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_redistribute(self, item, config=None):
    commands = list()
    safe_list = list()
    for entry in item['redistribute']:
        option = entry['protocol']
        cmd = 'redistribute %s' % entry['protocol']
        if entry['id'] and entry['protocol'] in ('ospf', 'table'):
            cmd += ' %s' % entry['id']
            option += ' %s' % entry['id']
        if entry['metric']:
            cmd += ' metric %s' % entry['metric']
        if entry['route_map']:
            cmd += ' route-map %s' % entry['route_map']
        if not config or cmd not in config:
            commands.append(cmd)
        safe_list.append(option)
    if self.params['operation'] == 'replace':
        if config:
            matches = re.findall('redistribute (\\S+)(?:\\s*)(\\d*)', config, re.M)
            for i in range(0, len(matches)):
                matches[i] = ' '.join(matches[i]).strip()
            for entry in set(matches).difference(safe_list):
                commands.append('no redistribute %s' % entry)
    return commands