from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_distance(config_data):
    command = 'distance bgp'
    if config_data['distance'].get('external'):
        command += ' {external}'.format(**config_data['distance'])
    if config_data['distance'].get('internal'):
        command += ' {internal}'.format(**config_data['distance'])
    if config_data['distance'].get('local'):
        command += ' {local}'.format(**config_data['distance'])
    return command