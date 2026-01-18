from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_default_information(proc):
    default_information = proc['default_information']['originate']
    command = 'default-information originate'
    if default_information.get('set') is False:
        command = 'no {0}'.format(command)
    else:
        if default_information.get('always'):
            command += ' always'
        if default_information.get('route_map'):
            command += ' route-map {route_map}'.format(**default_information)
    return command