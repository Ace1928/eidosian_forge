from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_allocate_label(config_data):
    if 'allocate_label' in config_data:
        command = 'allocate-label'
        if 'all' in config_data['allocate_label']:
            command += ' all'
        if 'route_policy' in config_data['allocate_label']:
            command += ' route-policy {route_policy}'.format(**config_data['route_policy'])
        return command