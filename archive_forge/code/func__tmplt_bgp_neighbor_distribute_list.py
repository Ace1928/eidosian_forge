from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_neighbor_distribute_list(config_data):
    command = []
    for list_el in config_data['neighbor']['distribute_list']:
        command.append('protocols bgp {as_number} '.format(**config_data) + 'neighbor {address} distribute-list '.format(**config_data['neighbor']) + list_el['action'] + ' ' + str(list_el['acl']))
    return command