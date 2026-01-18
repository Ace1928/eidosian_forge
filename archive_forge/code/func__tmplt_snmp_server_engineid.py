from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_engineid(config_data):
    command = []
    cmd = 'snmp-server engineID'
    el = config_data['engineid']
    if el.get('local'):
        c = cmd + ' local ' + el['local']
        command.append(c)
    if el.get('remote'):
        c = cmd + ' remote ' + el['remote']['host']
        if el['remote'].get('udp_port'):
            c += ' udp-port ' + str(el['remote']['udp_port'])
        if el['remote'].get('id'):
            c += ' ' + el['remote']['id']
            command.append(c)
    return command