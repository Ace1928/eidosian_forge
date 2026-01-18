from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_filter(config_data):
    if 'filter_list' in config_data:
        command = []
        for key, value in iteritems(config_data.get('filter_list')):
            cmd = 'area {area_id}'.format(**config_data)
            if value.get('name') and value.get('direction'):
                cmd += ' filter-list prefix {name} {direction}'.format(**value)
            command.append(cmd)
        return command