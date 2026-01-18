from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_ipv4_comm(config_data):
    command = ''
    if not config_data['communities'].get('acl_v6'):
        command = 'snmp-server community '
        el = config_data['communities']
        command += el['name']
        if el.get('view'):
            command += ' view ' + el['view']
        if el.get('ro'):
            command += ' ro'
        if el.get('rw'):
            command += ' rw'
        if el.get('acl_v4'):
            command += ' ' + el['acl_v4']
    return command