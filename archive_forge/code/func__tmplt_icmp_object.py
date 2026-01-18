from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_icmp_object(config_data):
    commands = []
    if config_data.get('icmp_type').get('icmp_object'):
        for each in config_data.get('icmp_type').get('icmp_object'):
            commands.append('icmp-object {0}'.format(each))
        return commands