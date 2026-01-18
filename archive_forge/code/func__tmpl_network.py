from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_network(config_data):
    cmd = 'network {network}'
    if config_data.get('backdoor_route_policy'):
        cmd += ' backdoor-route-policy {backdoor-route-policy}'
    if config_data.get('route_policy'):
        cmd += ' route-policy {route_policy}'
    return cmd.format(**config_data)