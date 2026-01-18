from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_maximum_paths(config_data):
    command = []
    for list_el in config_data['maximum_paths']:
        command.append('protocols bgp {as_number} maximum-paths '.format(**config_data) + list_el['path'] + ' ' + str(list_el['count']))
    return command