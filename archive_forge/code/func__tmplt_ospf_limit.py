from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_limit(config_data):
    if 'limit' in config_data:
        command = 'limit retransmissions'
        if 'dc' in config_data['limit']:
            if 'number' in config_data['limit']['dc']:
                command += ' dc {number}'.format(**config_data['limit']['dc'])
            if 'disable' in config_data['limit']['dc']:
                command += ' dc disable'
        if 'non_dc' in config_data['limit']:
            if 'number' in config_data['limit']['non_dc']:
                command += ' non-dc {number}'.format(**config_data['limit']['non_dc'])
            if 'disable' in config_data['limit']['dc']:
                command += ' non-dc disable'
        return command