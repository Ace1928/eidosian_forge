from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_set_ipv6(config_data):
    if config_data.get('set') and config_data['set'].get('ipv6'):
        set_ipv6 = config_data['set']['ipv6']
        cmd = 'set ipv6'
        if set_ipv6.get('address'):
            cmd += ' address prefix-list {address}'.format(**set_ipv6)
        if set_ipv6.get('default'):
            cmd += ' default'
        if set_ipv6.get('global_route'):
            cmd += ' global next-hop'
            if set_ipv6['global_route'].get('verify_availability'):
                cmd += ' verify-availability {address} {sequence} track {track}'.format(**set_ipv6['global_route']['verify_availability'])
            elif set_ipv6['global_route'].get('address'):
                cmd += ' {address}'.format(**set_ipv6['global_route'])
        if set_ipv6.get('next_hop'):
            cmd += ' next-hop'
            if set_ipv6['next_hop'].get('address'):
                cmd += ' {address}'.format(**set_ipv6['next_hop'])
            if set_ipv6['next_hop'].get('encapsulate'):
                cmd += ' encapsulate l3vpn {encapsulate}'.format(**set_ipv6['next_hop'])
            if set_ipv6['next_hop'].get('peer_address'):
                cmd += ' peer-address'
        if set_ipv6.get('precedence'):
            cmd += ' precedence {precedence}'.format(**set_ipv6)
        if set_ipv6.get('vrf'):
            cmd += ' vrf {vrf} next-hop verify-availability {address} {sequence} track {track}'.format(**set_ipv6['vrf']['verify_availability'])
        return cmd