from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_address_family(config_data):
    command = ''
    if config_data.get('vrf'):
        command = 'vrf {vrf}\n'.format(**config_data)
    command += 'address-family {afi}'.format(**config_data)
    if config_data.get('safi'):
        command += ' {safi}'.format(**config_data)
    return command