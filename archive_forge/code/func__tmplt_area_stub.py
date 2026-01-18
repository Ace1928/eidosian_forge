from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_area_stub(area):
    stub = area['stub']
    command = 'area {area_id} stub'.format(**area)
    if stub.get('set') is False:
        command = 'no {0}'.format(command)
    elif stub.get('no_summary'):
        command += ' no-summary'
    return command