from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_match_aggregator_role(config_data):
    config_data = config_data['entries']['match']['aggregator_role']
    command = 'match aggregator-role contributor'
    if config_data.get('route_map'):
        command += ' aggregate-attributes ' + config_data['route_map']
    return command