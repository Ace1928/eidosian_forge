from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_soft_reconfiguration(config_data):
    conf = config_data.get('soft_reconfiguration', {})
    if conf:
        command = 'soft-reconfiguration '
        if 'inbound' in conf:
            command += 'inbound'
            if 'set' in conf['inbound']:
                pass
            elif 'always' in conf['inbound']:
                command += ' always'
            if 'inheritance_disable' in conf['inbound']:
                command += ' inheritance-disable'
    return command