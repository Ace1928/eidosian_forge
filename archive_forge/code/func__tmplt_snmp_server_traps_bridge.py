from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_bridge(config_data):
    command = 'snmp-server enable traps bridge'
    el = config_data['traps']['bridge']
    if el.get('arista_mac_age'):
        command += ' arista-mac-age'
    if el.get('arista_mac_learn'):
        command += ' arista-mac-learn'
    if el.get('arista_mac_move'):
        command += ' arista-mac-move'
    return command