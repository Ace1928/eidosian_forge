from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_authentication_key(config_data):
    if 'authentication_key' in config_data:
        command = 'area {area_id} authentication-key'.format(**config_data)
        if config_data['authentication_key'].get('password'):
            command += ' {0}'.format(config_data['authentication_key'].get('password'))
        return command