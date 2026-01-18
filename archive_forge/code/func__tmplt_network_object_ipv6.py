from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_network_object_ipv6(config_data):
    commands = []
    if config_data.get('network_object').get('ipv6_address'):
        for each in config_data.get('network_object').get('ipv6_address'):
            commands.append('network-object {0}'.format(each))
        return commands