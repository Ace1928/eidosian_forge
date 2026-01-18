from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_params_distance(config_data):
    command = 'protocols bgp {as_number} parameters distance global '.format(**config_data) + config_data['bgp_params']['distance']['type'] + ' ' + str(config_data['bgp_params']['distance']['value'])
    return command