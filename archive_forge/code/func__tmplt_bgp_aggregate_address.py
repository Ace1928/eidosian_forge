from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_aggregate_address(config_data):
    command = 'protocols bgp {as_number} aggregate-address'.format(**config_data)
    if config_data['aggregate_address'].get('as_set'):
        command += ' {prefix} as-set'.format(**config_data['aggregate_address'])
    if config_data['aggregate_address'].get('summary_only'):
        command += ' {prefix} summary-only'.format(**config_data['aggregate_address'])
    return command