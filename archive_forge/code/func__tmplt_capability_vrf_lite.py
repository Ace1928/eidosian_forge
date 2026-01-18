from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_capability_vrf_lite(proc):
    command = 'capability vrf-lite'
    vrf_lite = proc['capability']['vrf_lite']
    if vrf_lite.get('set') is False:
        command = 'no {0}'.format(command)
    elif vrf_lite.get('evpn'):
        command += ' evpn'
    return command