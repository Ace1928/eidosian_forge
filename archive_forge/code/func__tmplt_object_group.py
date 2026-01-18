from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_object_group(config_data):
    command = 'object-group {object_type} {name}'.format(**config_data)
    return command