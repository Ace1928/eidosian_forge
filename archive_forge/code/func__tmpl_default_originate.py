from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_default_originate(config_data):
    conf = config_data.get('default_originate', {})
    command = ''
    if conf:
        if 'set' in conf:
            command = 'default-originate'
        if 'inheritance_disable' in conf:
            command = 'default-originate inheritance-disable'
        if 'route_policy' in conf:
            command = 'default-originate route_policy ' + conf['route_policy']
    return command