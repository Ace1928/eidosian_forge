from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_nsf_cisco(config_data):
    if 'cisco' in config_data['nsf']:
        command = 'nsf cisco helper'
        if 'disable' in config_data['nsf']['cisco']:
            command += ' disable'
        return command