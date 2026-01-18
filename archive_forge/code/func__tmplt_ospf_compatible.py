from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_compatible(config_data):
    if 'compatible' in config_data:
        if 'rfc1583' in config_data['compatible']:
            command = 'compatible rfc1583'
        elif 'rfc1587' in config_data['compatible']:
            command = 'compatible rfc1587'
        elif 'rfc5243' in config_data['compatible']:
            command = 'compatible rfc5243'
        return command