from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_access_list_name(config_data):
    command = 'access-list {acls_name} '.format(**config_data)
    return command