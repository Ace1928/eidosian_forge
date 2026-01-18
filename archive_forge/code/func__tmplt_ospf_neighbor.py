from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_neighbor(config_data):
    if 'neighbor' in config_data:
        command = 'neighbor'
        if 'address' in config_data['neighbor']:
            command += ' {address}'.format(**config_data['neighbor'])
        if 'cost' in config_data['neighbor']:
            command += ' cost {cost}'.format(**config_data['neighbor'])
        if 'database_filter' in config_data['neighbor']:
            command += ' database-filter all out'
        if 'poll_interval' in config_data['neighbor']:
            command += ' poll-interval {poll_interval}'.format(**config_data['neighbor'])
        if 'priority' in config_data['neighbor']:
            command += ' priority {priority}'.format(**config_data['neighbor'])
        return command