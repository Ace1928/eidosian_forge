from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_route_target(config_data):
    command = 'route-target {action}'.format(**config_data['route_target'])
    if config_data['route_target'].get('type'):
        command += ' {type}'.format(**config_data['route_target'])
    if config_data['route_target'].get('route_map'):
        command += ' {route_map}'.format(**config_data['route_target'])
    if config_data['route_target'].get('imported_route'):
        command += ' imported-route'
    if config_data['route_target'].get('target'):
        command += ' {target}'.format(**config_data['route_target'])
    return command