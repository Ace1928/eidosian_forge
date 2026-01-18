from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_cost_fallback_config(config_data):
    command = _compute_command(config_data)
    fallback = config_data['address_family']['cost_fallback']
    command += ' cost-fallback ' + str(fallback['cost']) + ' threshold ' + str(fallback['threshold'])
    return command