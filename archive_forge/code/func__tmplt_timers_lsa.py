from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_timers_lsa(config_data):
    if 'timers' in config_data:
        command = 'timers lsa'
        if 'group_pacing' in config_data['timers']['lsa']:
            command += ' group-pacing {group_pacing}'.format(**config_data['timers']['lsa'])
        if 'min_arrival' in config_data['timers']['lsa']:
            command += ' min-arrival {min_arrival}'.format(**config_data['timers']['lsa'])
        if 'refresh' in config_data['timers']['lsa']:
            command += ' refresh {refresh}'.format(**config_data['timers']['lsa'])
        return command