from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_delete_bgp_maximum_paths(config_data):
    command = 'protocols bgp {as_number} maximum-paths'.format(**config_data)
    return command