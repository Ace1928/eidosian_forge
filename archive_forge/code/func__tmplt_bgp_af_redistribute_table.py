from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_af_redistribute_table(config_data):
    if config_data['address_family']['redistribute'].get('table'):
        afi = config_data['address_family']['afi'] + '-unicast'
        command = 'protocols bgp {as_number} address-family '.format(**config_data)
        if config_data['address_family']['redistribute'].get('table'):
            command += afi + ' table {table}'.format(**config_data['address_family']['redistribute'])
        return command