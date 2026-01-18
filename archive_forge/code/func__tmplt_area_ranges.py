from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_area_ranges(arange):
    command = 'area {area_id} range {prefix}'.format(**arange)
    if arange.get('not_advertise') is True:
        command += ' not-advertise'
    if 'cost' in arange:
        command += ' cost {cost}'.format(**arange)
    return command