from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_int_authentication_key(config_data):
    if 'authentication_key' in config_data:
        command = 'ip ospf authentication-key'
        if 'encryption' in config_data['authentication_key']:
            command += ' {encryption} {key}'.format(**config_data['authentication_key'])
        else:
            command += ' {key}'.format(**config_data['authentication_key'])
        return command