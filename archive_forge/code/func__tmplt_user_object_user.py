from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_user_object_user(config_data):
    commands = []
    if config_data.get('user_object').get('user'):
        for each in config_data.get('user_object').get('user'):
            commands.append('user {domain}\\{name}'.format(**each))
    return commands