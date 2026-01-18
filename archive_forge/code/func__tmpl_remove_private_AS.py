from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_remove_private_AS(config_data):
    conf = config_data.get('remove_private_AS', {})
    if conf:
        command = ' '
        if 'set' in conf:
            command = 'remove-private-AS'
        if 'inbound' in conf:
            command += ' inbound'
        if 'entire_aspath' in conf:
            command += ' entire-aspath'
        elif 'inheritance_disable' in conf:
            command = 'remove-private-AS inheritance-disable'
    return command