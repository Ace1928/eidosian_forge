from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_prefix_suppression(config_data):
    if 'prefix_suppression' in config_data:
        if 'set' in config_data['prefix_suppression']:
            command = 'prefix-suppression'
        if 'secondary_address' in config_data['prefix_suppression']:
            command = 'prefix-suppression secondary-address'
        return command