from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_network_object_object(config_data):
    commands = []
    if config_data.get('network_object').get('object'):
        for each in config_data.get('network_object').get('object'):
            commands.append('network-object object {0}'.format(each))
        return commands