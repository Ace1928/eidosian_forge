from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_distance_ospf(config_data):
    if 'ospf' in config_data['distance']:
        command = 'distance ospf'
        if 'inter_area' in config_data['distance']['ospf']:
            command += ' inter-area {inter_area}'.format(**config_data['distance']['ospf'])
        if config_data['distance'].get('ospf').get('intra_area'):
            command += ' intra-area {intra_area}'.format(**config_data['distance']['ospf'])
        if config_data['distance'].get('ospf').get('external'):
            command += ' external {external}'.format(**config_data['distance']['ospf'])
        return command