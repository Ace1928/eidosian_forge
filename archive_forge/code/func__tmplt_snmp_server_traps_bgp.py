from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_bgp(config_data):
    command = 'snmp-server enable traps bgp'
    el = config_data['traps']['bgp']
    if el.get('arista_backward_transition'):
        command += ' arista-backward-transition'
    if el.get('arista_established'):
        command += ' arista-established'
    if el.get('backward_transition'):
        command += ' backward-transition'
    if el.get('established'):
        command += ' established'
    return command