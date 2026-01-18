from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_users_localized(config_data):
    el = config_data['users']
    command = 'snmp-server user ' + el['user'] + ' ' + el['group']
    if el.get('remote'):
        command += ' remote ' + el['remote']
    if el.get('udp_port'):
        command += ' udp-port ' + str(el['udp_port'])
    command += ' ' + el['version']
    if el.get('localized'):
        command += ' localized ' + el['localized']['engineid']
        el = el['localized']
        command += ' ' + el['algorithm'] + ' ' + el['auth_passphrase']
        if el.get('encryption'):
            command += ' priv ' + el['encryption'] + ' ' + el['priv_passphrase']
    return command