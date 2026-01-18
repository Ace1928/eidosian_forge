from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_area_nssa(area):
    nssa = area['nssa']
    command = 'area {area_id} nssa'.format(**area)
    if nssa.get('set') is False:
        command = 'no {0}'.format(command)
    else:
        for attrib in ['no_summary', 'no_redistribution', 'default_information_originate']:
            if nssa.get(attrib):
                command += ' {0}'.format(attrib.replace('_', '-'))
    return command