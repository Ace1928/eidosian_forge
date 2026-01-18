from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_prefix_list_ip_del(config_data):
    command_set = []
    config_data = config_data['prefix_lists'].get('entries', {})
    for k, v in iteritems(config_data):
        command_set.append('seq ' + str(k))
    return command_set