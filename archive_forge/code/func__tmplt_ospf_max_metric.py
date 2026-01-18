from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_max_metric(config_data):
    if 'max_metric' in config_data:
        command = 'max-metric'
        if 'router_lsa' in config_data['max_metric']:
            command += ' router-lsa'
        if 'external_lsa' in config_data['max_metric']:
            command += ' external-lsa {external_lsa}'.format(**config_data['max_metric'])
        if 'include_stub' in config_data['max_metric']:
            command += ' include-stub'
        if 'on_startup' in config_data['max_metric']:
            if 'time' in config_data['max_metric']['on_startup']:
                command += ' on-startup {time}'.format(**config_data['max_metric']['on_startup'])
            elif 'wait_for_bgp' in config_data['max_metric']['on_startup']:
                command += ' on-startup wait-for-bgp'
        if 'summary_lsa' in config_data['max_metric']:
            command += ' summary-lsa {summary_lsa}'.format(**config_data['max_metric'])
        return command