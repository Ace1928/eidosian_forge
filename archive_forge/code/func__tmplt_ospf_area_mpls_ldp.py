from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_mpls_ldp(config_data):
    commands = []
    if 'mpls' in config_data:
        command = 'area {area_id} mpls'.format(**config_data)
        if config_data['mpls'].get('ldp'):
            ldp = config_data['mpls'].get('ldp')
            if 'auto_config' in ldp:
                command += ' auto-config'
                commands.append(command)
            if 'sync' in ldp:
                command += ' sync'
                commands.append(command)
            if 'sync_igp_shortcuts' in ldp:
                command += ' sync-igp-shortcuts'
                commands.append(command)
        return commands