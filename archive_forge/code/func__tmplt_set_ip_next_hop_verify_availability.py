from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_set_ip_next_hop_verify_availability(data):
    cmd = []
    for each in data['set']['ip']['next_hop']['verify_availability']:
        cmd_tmpl = 'set ip next-hop verify-availability'
        cmd_tmpl += ' {address} track {track}'.format(**each)
        if 'load_share' in each and each['load_share']:
            cmd_tmpl += ' load-share'
        if 'force_order' in each and each['force_order']:
            cmd_tmpl += ' force-order'
        if 'drop_on_fail' in each and each['drop_on_fail']:
            cmd_tmpl += ' drop-on-fail'
        cmd.append(cmd_tmpl)
    return cmd