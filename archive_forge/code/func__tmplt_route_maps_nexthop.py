from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_maps_nexthop(config_data):
    config_data = config_data['entries']['set']['nexthop']
    command = 'set next-hop igp-metric '
    if config_data.get('max_metric'):
        command += 'max-metric'
    if config_data.get('value'):
        command += config_data['value']
    return command