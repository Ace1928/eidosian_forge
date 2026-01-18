from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_delete_redistribute(config_data):
    command = 'protocols bgp {as_number} redistribute '.format(**config_data) + config_data['redistribute']['protocol']
    return command