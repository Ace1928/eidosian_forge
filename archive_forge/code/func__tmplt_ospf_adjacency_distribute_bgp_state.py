from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_adjacency_distribute_bgp_state(config_data):
    if 'distribute_link_list' in config_data:
        command = 'distribute link-state'
        if config_data['distribute_link_list'].get('instance_id'):
            command += '  instance-id {0}'.format(config_data['distribute_link_list'].get('instance_id'))
        elif config_data['distribute_link_list'].get('throttle'):
            command += '  throttle {0}'.format(config_data['distribute_link_list'].get('throttle'))
        return command
    elif 'distribute_bgp_ls' in config_data:
        command = 'distribute bgp-ls'
        if config_data['distribute_bgp_ls'].get('instance_id'):
            command += '  instance-id {0}'.format(config_data['distribute_bgp_ls'].get('instance_id'))
        elif config_data['distribute_bgp_ls'].get('throttle'):
            command += '  throttle {0}'.format(config_data['distribute_bgp_ls'].get('throttle'))
        return command