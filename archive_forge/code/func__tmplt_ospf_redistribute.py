from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_redistribute(config_data):
    command = 'redistribute {routes}'.format(**config_data)
    if 'route_map' in config_data:
        command += ' route-map {route_map}'.format(**config_data)
    return command