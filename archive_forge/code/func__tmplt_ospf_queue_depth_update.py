from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_queue_depth_update(config_data):
    if 'update' in config_data['queue_depth']:
        command = 'queue-depth update'
        if 'max_packets' in config_data['queue_depth']['update']:
            command += ' {max_packets}'.format(**config_data['queue_depth']['update'])
        elif 'unlimited' in config_data['queue_depth']['update']:
            command += ' unlimited'
        return command