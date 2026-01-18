from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_nsr(config_data):
    if 'nsr' in config_data:
        command = 'nsr'
        if 'set' in config_data['nsr']:
            command += ' nsr'
        elif config_data['nsr'].get('disable'):
            command += ' nsr {0}'.format('disable')
        return command