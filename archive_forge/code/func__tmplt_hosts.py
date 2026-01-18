from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_hosts(data):
    cmd = 'logging server {host}'
    data['client_identity'] = data.get('secure', {}).get('trustpoint', {}).get('client_identity')
    if 'severity' in data:
        cmd += ' {severity}'
    if 'port' in data:
        cmd += ' port {port}'
    if data['client_identity']:
        cmd += ' secure trustpoint client-identity {client_identity}'
    if 'facility' in data:
        cmd += ' facility {facility}'
    if 'use_vrf' in data:
        cmd += ' use-vrf {use_vrf}'
    cmd = cmd.format(**data)
    return cmd