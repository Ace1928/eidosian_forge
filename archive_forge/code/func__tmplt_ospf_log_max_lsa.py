from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_log_max_lsa(config_data):
    if 'max_lsa' in config_data:
        command = 'max-lsa'
        if 'threshold' in config_data['max_lsa']:
            command += ' {0}'.format(config_data['max_lsa'].get('threshold'))
        if 'warning_only' in config_data['max_lsa']:
            command += ' warning-only {0}'.format(config_data['max_lsa'].get('warning_only'))
        if 'ignore_time' in config_data['max_lsa']:
            command += ' ignore-time {0}'.format(config_data['max_lsa'].get('ignore_time'))
        if 'ignore_count' in config_data['max_lsa']:
            command += ' ignore-count {0}'.format(config_data['max_lsa'].get('ignore_count'))
        if 'reset_time' in config_data['max_lsa']:
            command += ' reset-time {0}'.format(config_data['max_lsa'].get('reset_time'))
        return command