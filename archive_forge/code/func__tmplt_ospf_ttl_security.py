from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_ttl_security(config_data):
    if 'ttl_security' in config_data:
        command = 'ttl-security all-interfaces'
        if 'hops' in config_data['ttl_security']:
            command += ' hops {hops}'.format(**config_data['ttl_security'])
        return command