from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_ospfv3(config_data):
    command = 'snmp-server enable traps ospfv3'
    el = config_data['traps']['ospfv3']
    if el.get('if_config_error'):
        command += ' if-config-error'
    if el.get('if_rx_bad_packet'):
        command += ' if-rx-bad-packet'
    if el.get('if_state_change'):
        command += ' if-state-change'
    if el.get('nbr_state_change'):
        command += ' nbr-state-change'
    if el.get('nbr_restart_helper_status_change'):
        command += ' nbr-restart-helper-status-change'
    if el.get('nssa_translator_status_change'):
        command += ' nssa-translator-status-change'
    if el.get('restart_status_change'):
        command += ' restart-status-change'
    return command