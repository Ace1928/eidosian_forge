from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_neighbor_timers(config_data):
    command = []
    for k, v in iteritems(config_data['neighbor']['timers']):
        command.append('protocols bgp {as_number} neighbor '.format(**config_data) + config_data['neighbor']['address'] + ' timers ' + k + ' ' + str(v))
    return command