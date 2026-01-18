from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_sham_link(config_data):
    if 'sham_link' in config_data:
        command = 'area {area_id} sham-link'.format(**config_data)
        if 'source' in config_data['sham_link']:
            command += ' {source} {destination}'.format(**config_data['sham_link'])
        if 'cost' in config_data['sham_link']:
            command += ' cost {cost}'.format(**config_data['sham_link'])
        if 'ttl_security' in config_data['sham_link']:
            command += ' ttl-security hops {ttl_security}'.format(**config_data['sham_link'])
        return command