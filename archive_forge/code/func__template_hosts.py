from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _template_hosts(data):
    cmd = 'snmp-server host {0}'.format(data['host'])
    if data.get('traps'):
        cmd += ' traps'
    if data.get('informs'):
        cmd += ' informs'
    if data.get('use_vrf'):
        cmd += ' use-vrf {0}'.format(data['use_vrf'])
    if data.get('filter_vrf'):
        cmd += ' filter-vrf {0}'.format(data['filter_vrf'])
    if data.get('source_interface'):
        cmd += ' source-interface {0}'.format(data['source_interface'])
    if data.get('version'):
        cmd += ' version {0}'.format(data['version'])
    if data.get('community'):
        cmd += ' ' + data['community']
    elif data.get('auth'):
        cmd += ' auth {0}'.format(data['auth'])
    elif data.get('priv'):
        cmd += ' priv {0}'.format(data['priv'])
    if data.get('udp_port'):
        cmd += ' udp-port {0}'.format(data['udp_port'])
    return cmd