from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_acls(config_data):
    command = 'snmp-server ' + config_data['acls']['afi'] + ' access-list '
    el = config_data['acls']
    command += el['acl']
    if el.get('vrf'):
        command += ' vrf ' + el['vrf']
    return command