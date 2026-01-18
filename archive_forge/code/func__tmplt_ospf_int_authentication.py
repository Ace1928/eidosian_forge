from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_int_authentication(config_data):
    if 'authentication_v2' in config_data:
        command = 'ip ospf authentication'
        if 'message_digest' in config_data['authentication_v2']:
            command += ' message-digest'
        return command
    if 'authentication_v3' in config_data:
        command = 'ospfv3 authentication ipsec spi '
        command += '{spi} {algorithm}'.format(**config_data['authentication_v3'])
        if 'passphrase' in config_data['authentication_v3']:
            command += ' passphrase'
        if 'keytype' in config_data['authentication_v3']:
            command += ' {keytype}'.format(**config_data['authentication_v3'])
        if 'passphrase' not in config_data['authentication_v3']:
            command += ' {key}'.format(**config_data['authentication_v3'])
        else:
            command += ' {passphrase}'.format(**config_data['authentication_v3'])
        return command