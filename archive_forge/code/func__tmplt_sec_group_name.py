from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_sec_group_name(config_data):
    commands = []
    if config_data.get('security_group').get('sec_name'):
        for each in config_data.get('security_group').get('sec_name'):
            commands.append('security-group name {0}'.format(each))
        return commands