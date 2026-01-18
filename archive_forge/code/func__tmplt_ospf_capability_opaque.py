from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_capability_opaque(config_data):
    if 'capability' in config_data:
        if 'opaque' in config_data['capability']:
            command = 'capability opaque'
            opaque = config_data['capability'].get('opaque')
            if 'disable' in opaque:
                command += 'capability opaque disable'
        return command