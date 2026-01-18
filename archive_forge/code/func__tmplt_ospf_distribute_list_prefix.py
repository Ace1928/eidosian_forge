from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_distribute_list_prefix(config_data):
    if 'prefix' in config_data.get('distribute_list'):
        command = 'distribute-list prefix {name}'.format(**config_data['distribute_list']['prefix'])
        if 'gateway_name' in config_data['distribute_list']['prefix']:
            command += ' gateway {gateway_name}'.format(**config_data['distribute_list']['prefix'])
        if 'direction' in config_data['distribute_list']['prefix']:
            command += ' {direction}'.format(**config_data['distribute_list']['prefix'])
        if 'interface' in config_data['distribute_list']['prefix']:
            command += ' {interface}'.format(**config_data['distribute_list']['prefix'])
        if 'protocol' in config_data['distribute_list']['prefix']:
            command += ' {protocol}'.format(**config_data['distribute_list']['prefix'])
        return command