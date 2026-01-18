from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_distance_admin(config_data):
    if 'admin_distance' in config_data:
        command = 'distance'
        if config_data['admin_distance'].get('value'):
            command += ' {0}'.format(config_data['admin_distance'].get('value'))
        if config_data['admin_distance'].get('source'):
            command += ' {0}'.format(config_data['admin_distance'].get('source'))
        if config_data['admin_distance'].get('wildcard'):
            command += ' {0}'.format(config_data['admin_distance'].get('wildcard'))
        if config_data['admin_distance'].get('access_list'):
            command += ' {0}'.format(config_data['admin_distance'].get('access_list'))
        return command