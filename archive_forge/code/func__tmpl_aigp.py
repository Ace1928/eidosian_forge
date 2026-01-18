from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_aigp(config_data):
    conf = config_data.get('aigp', {})
    commands = []
    if conf:
        if 'set' in conf:
            commands.append('aigp')
        if 'disable' in conf:
            commands.append('aigp disable')
        if 'send_cost_community_disable' in conf:
            commands.append('aigp send cost-community disable')
        if 'send_med' in conf and 'set' in conf.get('send_med', {}):
            commands.append('aigp send med')
        if 'send_med' in conf and 'disable' in conf.get('send_med', {}):
            commands.append('aigp send med disable')
    return commands