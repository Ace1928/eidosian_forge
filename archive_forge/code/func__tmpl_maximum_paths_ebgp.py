from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_maximum_paths_ebgp(config_data):
    ebgp_conf = config_data.get('maximum_paths', {}).get('ebgp', {})
    if ebgp_conf:
        command = 'maximum-paths ebgp'
        if 'max_path_value' in ebgp_conf:
            command += ' ' + str(ebgp_conf['max_path_value'])
        if 'order_igp_metric' in ebgp_conf:
            command += ' order igp-metric'
        elif 'selective_order_igp_metric' in ebgp_conf:
            command += ' selective order igp-metric'
        return command