from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_next_hop_unchanged(config_data):
    conf = config_data.get('next_hop_unchanged', {})
    command = ''
    if conf:
        if 'set' in conf:
            command = 'next-hop-unchanged'
        if 'inheritance_disable' in conf:
            command += 'next-hop-unchanged inheritance-disable'
        if 'multipath' in conf:
            command = 'next-hop-unchanged multipath'
    return command