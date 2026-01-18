from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_set_aspath_match(config_data):
    el = config_data['entries']
    command = 'set as-path match all replacement '
    c = el['set']['as_path']['match']
    if c.get('none'):
        command += 'none'
    if c.get('as_number'):
        num = str(c['as_number'])
        command += num
    return command