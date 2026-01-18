from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_adjacency_cmd(config_data):
    if 'adjacency' in config_data:
        command = 'adjacency stagger'
        if 'none' in config_data['adjacency']:
            command += ' none'
        else:
            command += ' {min_adjacency}'.format(**config_data['adjacency'])
        if 'max_adjacency' in config_data['adjacency']:
            command += ' {min_adjacency}'.format(**config_data['adjacency'])
        return command