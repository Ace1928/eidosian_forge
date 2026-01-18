from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.utils.utils import (
def _tmplt_ntp_global_servers(config_data):
    el = config_data['servers']
    command = 'ntp server'
    if el.get('vrf'):
        command += ' vrf {vrf}'.format(**el)
    if el.get('server'):
        command += ' {server}'.format(**el)
    if el.get('burst'):
        command += ' burst'
    if el.get('iburst'):
        command += ' iburst'
    if el.get('key_id'):
        command += ' key {key_id}'.format(**el)
    if el.get('local_interface'):
        linterface = el.get('local_interface').replace(' ', '')
        command += ' local-interface ' + linterface
    if el.get('maxpoll'):
        command += ' maxpoll {maxpoll}'.format(**el)
    if el.get('minpoll'):
        command += ' minpoll {minpoll}'.format(**el)
    if el.get('prefer'):
        command += ' prefer'
    if el.get('source'):
        interface_name = normalize_interface(el['source'])
        command += ' source ' + interface_name
    if el.get('version'):
        command += ' version {version}'.format(**el)
    return command