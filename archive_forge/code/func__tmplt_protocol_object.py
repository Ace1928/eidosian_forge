from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_protocol_object(config_data):
    commands = []
    if config_data.get('protocol_object').get('protocol'):
        for each in config_data.get('protocol_object').get('protocol'):
            commands.append('protocol {0}'.format(each))
        return commands