from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_path_attribute(proc):
    cmd = 'path-attribute {action}'.format(**proc)
    if 'type' in proc:
        cmd += ' {type}'.format(**proc)
    elif 'range' in proc:
        cmd += ' range {start} {end}'.format(**proc['range'])
    cmd += ' in'
    return cmd