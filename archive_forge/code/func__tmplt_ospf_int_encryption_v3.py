from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_int_encryption_v3(config_data):
    if 'encryption' in config_data:
        command = 'ospfv3 encryption ipsec spi '.format(**config_data)
        command += '{spi} esp {encryption} {algorithm}'.format(**config_data['encryption'])
        if 'passphrase' in config_data['encryption']:
            command += ' passphrase'
        if 'keytype' in config_data['encryption']:
            command += ' {keytype}'.format(**config_data['encryption'])
        if 'passphrase' not in config_data['encryption']:
            command += ' {key}'.format(**config_data['encryption'])
        else:
            command += ' {passphrase}'.format(**config_data['encryption'])
        return command