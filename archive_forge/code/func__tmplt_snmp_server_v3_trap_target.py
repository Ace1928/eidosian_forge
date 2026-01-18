from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_v3_trap_target(config_data):
    config_data = config_data['snmp_v3']['trap_targets']
    command = 'service snmp v3 trap-target {address} '.format(**config_data)
    if 'authentication' in config_data:
        command += ' auth'
        config_data = config_data['authentication']
    if 'privacy' in config_data:
        command += ' privacy'
        config_data = config_data['privacy']
    if 'type' in config_data:
        command += ' type {mode}'.format(**config_data)
    if 'encrypted_key' in config_data:
        command += ' encrypted-key {encrypted_key}'.format(**config_data)
    if 'plaintext_key' in config_data:
        command += ' plaintext-key {plaintext_key}'.format(**config_data)
    return command