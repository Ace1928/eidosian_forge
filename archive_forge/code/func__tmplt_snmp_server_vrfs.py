from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_vrfs(config_data):
    command = 'snmp-server vrf ' + config_data['vrfs']['vrf']
    el = config_data['vrfs']
    if el.get('local_interface'):
        command += ' local-interface ' + el['local_interface']
    return command