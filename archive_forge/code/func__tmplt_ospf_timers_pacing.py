from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_timers_pacing(config_data):
    if 'pacing' in config_data['timers']:
        command = 'timers pacing'
        if 'flood' in config_data['timers']['pacing']:
            command += ' flood {flood}'.format(**config_data['timers']['pacing'])
        elif 'lsa_group' in config_data['timers']['pacing']:
            command += ' lsa-group {lsa_group}'.format(**config_data['timers']['pacing'])
        elif 'retransmission' in config_data['timers']['pacing']:
            command += ' retransmission {retransmission}'.format(**config_data['timers']['pacing'])
        return command