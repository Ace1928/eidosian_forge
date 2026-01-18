from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_match_aspath(config_data):
    config_data = config_data['entries']['match']['as_path']
    command = 'match as-path '
    if config_data.get('length'):
        command += 'length ' + config_data['length']
    if config_data.get('path_list'):
        command += config_data['path_list']
    return command