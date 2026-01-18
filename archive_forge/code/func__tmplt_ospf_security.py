from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_security(config_data):
    if 'security_ttl' in config_data:
        command = 'security_ttl'
        if 'set' in config_data['security_ttl']:
            command += ' ttl'
        elif config_data['security_ttl'].get('hops'):
            command += ' ttl hops {0}'.format(config_data['security_ttl'].get('hops'))
        return command