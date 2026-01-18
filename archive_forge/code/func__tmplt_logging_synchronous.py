from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_logging_synchronous(config_data):
    command = 'logging synchronous'
    if 'level' in config_data['synchronous']:
        command += ' level ' + config_data['synchronous']['level']
    return command