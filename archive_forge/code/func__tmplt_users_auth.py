from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_users_auth(data):
    cmd = 'snmp-server user {0}'.format(data['user'])
    if 'group' in data:
        cmd += ' {0}'.format(data['group'])
    if 'authentication' in data:
        auth = data['authentication']
        if 'algorithm' in auth:
            cmd += ' auth {0}'.format(auth['algorithm'])
        if 'password' in auth:
            cmd += ' {0}'.format(auth['password'])
        priv = auth.get('priv', {})
        if priv:
            cmd += ' priv'
            if priv.get('aes_128', False):
                cmd += ' aes-128'
            if 'privacy_password' in priv:
                cmd += ' {0}'.format(priv['privacy_password'])
        if auth.get('localized_key', False):
            cmd += ' localizedkey'
        elif auth.get('localizedv2_key', False):
            cmd += ' localizedV2key'
        if 'engine_id' in auth:
            cmd += ' engineID {0}'.format(auth['engine_id'])
        return cmd