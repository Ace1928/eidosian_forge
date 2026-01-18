from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_distribute_list_acls(config_data):
    if 'acls' in config_data.get('distribute_list'):
        command = []
        for k, v in iteritems(config_data['distribute_list']['acls']):
            cmd = 'distribute-list {name} {direction}'.format(**v)
            if 'interface' in v:
                cmd += ' {interface}'.format(**v)
            if 'protocol' in v:
                cmd += ' {protocol}'.format(**v)
            command.append(cmd)
        return command