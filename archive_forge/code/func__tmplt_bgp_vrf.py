from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_vrf(config_data):
    command = 'vrf {vrf}'.format(**config_data)
    return command