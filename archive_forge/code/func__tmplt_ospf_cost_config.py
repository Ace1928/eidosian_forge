from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_cost_config(config_data):
    command = _compute_command(config_data)
    command += ' cost ' + str(config_data['address_family']['cost'])
    return command