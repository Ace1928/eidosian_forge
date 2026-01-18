from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_logging_format(config_data):
    command = ''
    if 'hostname' in config_data['format']:
        command = 'logging format hostname ' + config_data['format']['hostname']
    if 'sequence_numbers' in config_data['format']:
        command = 'logging format sequence-numbers'
    return command