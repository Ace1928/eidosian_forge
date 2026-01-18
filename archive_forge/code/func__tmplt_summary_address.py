from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_summary_address(proc):
    command = 'summary-address {prefix}'.format(**proc)
    if proc.get('tag'):
        command += ' tag {tag}'.format(**proc)
    elif proc.get('not_advertise'):
        command += ' not-advertise'
    return command