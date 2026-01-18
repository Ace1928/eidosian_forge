from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_ucmp(config_data):
    command = 'ucmp'
    if 'fec' in config_data['ucmp']:
        command += ' fec threshold trigger'
        command += ' {trigger} clear {clear} warning-only'.format(**config_data['ucmp']['fec'])
    if 'link_bandwidth' in config_data['ucmp']:
        command += ' link-bandwidth {mode}'.format(**config_data['ucmp']['link_bandwidth'])
        if config_data['ucmp']['link_bandwidth'].get('mode') == 'update_delay':
            command += ' {update_delay}'.format(**config_data['ucmp']['link_bandwidth'])
    if 'mode' in config_data['ucmp']:
        command += ' mode 1'
        if config_data['ucmp']['mode'].get('nexthops'):
            command += ' {nexthops}'.format(**config_data['ucmp']['mode'])
    return command