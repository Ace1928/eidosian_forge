from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_timers(timers_list, os_version='4.20'):
    timers_cmd = []
    for t_dict in timers_list:
        t_cmd = 'timers '
        for t_key in t_dict.keys():
            if not t_dict.get(t_key):
                break
            if t_key == 'lsa':
                if t_dict['lsa'].get('rx'):
                    if os_version < '4.23':
                        t_cmd = t_cmd + 'lsa arrival ' + str(t_dict['lsa']['rx']['min_interval'])
                    else:
                        t_cmd = t_cmd + 'lsa rx min interval ' + str(t_dict['lsa']['rx']['min_interval'])
                else:
                    t_cmd = t_cmd + 'lsa tx delay initial ' + str(t_dict['lsa']['tx']['delay']['initial']) + ' ' + str(t_dict['lsa']['tx']['delay']['min']) + ' ' + str(t_dict['lsa']['tx']['delay']['max'])
            elif t_key == 'out_delay':
                t_cmd = t_cmd + ' out-delay ' + str(t_dict['out_delay'])
            elif t_key == 'pacing':
                t_cmd = t_cmd + ' pacing flood ' + str(t_dict['pacing'])
            elif t_key == 'spf':
                if 'seconds' in t_dict['spf'].keys():
                    t_cmd = t_cmd + ' spf ' + str(t_dict['spf']['seconds'])
                else:
                    t_cmd = t_cmd + 'spf delay initial ' + str(t_dict['spf']['initial']) + ' ' + str(t_dict['spf']['max']) + ' ' + str(t_dict['spf']['min'])
            elif t_key == 'throttle':
                if t_dict['throttle']['attr'] == 'lsa':
                    t_cmd = t_cmd + 'throttle lsa all '
                else:
                    t_cmd = t_cmd + 'throttle spf '
                t_cmd = t_cmd + str(t_dict['throttle']['initial']) + ' ' + str(t_dict['throttle']['min']) + ' ' + str(t_dict['throttle']['max'])
            timers_cmd.append(t_cmd)
    return timers_cmd