from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_ospf(config_data):
    command = 'snmp-server enable traps ospf'
    el = config_data['traps']['ospf']
    if el.get('if_auth_failure'):
        command += ' if-auth-failure'
    if el.get('if_config_error'):
        command += ' if-config-error'
    if el.get('if_state_change'):
        command += ' if-state-change'
    if el.get('nbr_state_change'):
        command += ' nbr-state-change'
    return command