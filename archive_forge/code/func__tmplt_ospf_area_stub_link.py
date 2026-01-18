from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_stub_link(config_data):
    if 'stub' in config_data:
        command = 'area {area_id} stub'.format(**config_data)
        if 'no_ext_capability' in config_data['stub']:
            command += ' no-ext-capability'
        if 'no_summary' in config_data['stub']:
            command += ' no-summary'
        return command