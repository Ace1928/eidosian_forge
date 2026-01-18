from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_authentication_md(config_data):
    if 'authentication' in config_data:
        command = 'area {area_id} authentication'.format(**config_data)
        if 'message_digest' in config_data['authentication']:
            command = 'authentication message-digest'
            md = config_data['authentication'].get('message_digest')
            if md.get('keychain'):
                command += ' keychain ' + md.get('keychain')
        return command