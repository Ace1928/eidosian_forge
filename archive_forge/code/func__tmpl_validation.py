from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_validation(config_data):
    conf = config_data.get('validation', {})
    command = ''
    if conf:
        if 'set' in conf:
            command = 'validation'
        if 'disable' in conf:
            command = 'validation disbale'
        if 'redirect' in conf:
            command = 'validation redirect'
    return command