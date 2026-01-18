from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_mpls_ldp(config_data):
    command = 'snmp-server enable traps mpls-ldp'
    el = config_data['traps']['mpls_ldp']
    if el.get('mpls_ldp_session_down'):
        command += ' mpls-ldp-session-down'
    if el.get('mpls_ldp_session_up'):
        command += ' mpls-ldp-session-up'
    return command