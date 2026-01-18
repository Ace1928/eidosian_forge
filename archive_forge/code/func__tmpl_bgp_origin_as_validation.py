from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_bgp_origin_as_validation(config_data):
    origin_as_conf = config_data.get('bgp', {}).get('origin_as', {}).get('validation')
    if origin_as_conf:
        command = []
        if 'disable' in origin_as_conf:
            command.append('bgp origin-as validation disable')
        if 'ibgp' in origin_as_conf.get('signal', {}):
            command.append('bgp origin-as validation signal ibgp')
        return command