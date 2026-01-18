from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_area_nssa_translate(area):
    translate = area['nssa']['translate']['type7']
    command = 'area {area_id} nssa translate type7'.format(**area)
    for attrib in ['always', 'never', 'supress_fa']:
        if translate.get(attrib):
            command += ' {0}'.format(attrib.replace('_', '-'))
    return command