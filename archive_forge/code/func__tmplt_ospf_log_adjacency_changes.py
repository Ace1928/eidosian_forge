from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_log_adjacency_changes(config_data):
    if 'log_adjacency_changes' in config_data:
        command = 'log-adjacency-changes'
        if 'detail' in config_data['log_adjacency_changes']:
            command += ' detail'
        return command