from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_af_aggregate_address(config_data):
    afi = config_data['address_family']['afi'] + '-unicast'
    command = 'protocols bgp {as_number} address-family '.format(**config_data)
    config_data = config_data['address_family']
    if config_data['aggregate_address'].get('as_set'):
        command += afi + ' aggregate-address {prefix} as-set'.format(**config_data['aggregate_address'])
    if config_data['aggregate_address'].get('summary_only'):
        command += afi + ' aggregate-address {prefix} summary-only'.format(**config_data['aggregate_address'])
    return command