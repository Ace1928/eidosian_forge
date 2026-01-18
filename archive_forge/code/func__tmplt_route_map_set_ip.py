from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_set_ip(config_data):
    if config_data.get('set') and config_data['set'].get('ip'):
        command = []
        set_ip = config_data['set']['ip']
        cmd = 'set ip'
        if set_ip.get('address'):
            command.append('{0} address prefix-list {address}'.format(cmd, **set_ip))
        if set_ip.get('df'):
            command.append('{0} df {df}'.format(cmd, **set_ip))
        if set_ip.get('global_route'):
            cmd += ' global next-hop'
            if set_ip['global_route'].get('verify_availability'):
                cmd += ' verify-availability {address} {sequence} track {track}'.format(**set_ip['global_route']['verify_availability'])
            elif set_ip['global_route'].get('address'):
                cmd += ' {address}'.format(**set_ip['global_route'])
            command.append(cmd)
        if set_ip.get('next_hop'):
            cmd += ' next-hop'
            if set_ip['next_hop'].get('address'):
                command.append('{0} {address}'.format(cmd, **set_ip['next_hop']))
            if set_ip['next_hop'].get('dynamic'):
                command.append('{0} dynamic dhcp'.format(cmd))
            if set_ip['next_hop'].get('encapsulate'):
                command.append('{0} encapsulate l3vpn {encapsulate}'.format(cmd, **set_ip['next_hop']))
            if set_ip['next_hop'].get('peer_address'):
                command.append('{0} peer-address'.format(cmd))
            if set_ip['next_hop'].get('recursive'):
                child_cmd = '{0} recursive'.format(cmd)
                if set_ip['next_hop']['recursive'].get('global_route'):
                    child_cmd += ' global'
                elif set_ip['next_hop']['recursive'].get('vrf'):
                    child_cmd += ' vrf {vrf}'.format(**set_ip['next_hop']['recursive'])
                if set_ip['next_hop']['recursive'].get('address'):
                    child_cmd += ' {address}'.format(**set_ip['next_hop']['recursive'])
                command.append(child_cmd)
            if set_ip['next_hop'].get('self'):
                command.append('{0} self'.format(cmd))
            if set_ip['next_hop'].get('verify_availability'):
                command.append('{0} verify-availability {address} {sequence} track {track}'.format(cmd, **set_ip['next_hop']['verify_availability']))
        if set_ip.get('precedence'):
            cmd += ' precedence'
            if set_ip['precedence'].get('critical'):
                cmd += ' critical'
            elif set_ip['precedence'].get('flash'):
                cmd += ' flash'
            elif set_ip['precedence'].get('flash_override'):
                cmd += ' flash-override'
            elif set_ip['precedence'].get('immediate'):
                cmd += ' immediate'
            elif set_ip['precedence'].get('internet'):
                cmd += ' internet'
            elif set_ip['precedence'].get('network'):
                cmd += ' network'
            elif set_ip['precedence'].get('priority'):
                cmd += ' priority'
            elif set_ip['precedence'].get('routine'):
                cmd += ' routine'
            command.append(cmd)
        if set_ip.get('qos_group'):
            command.append('{0} qos-group {qos_group}'.format(cmd, **set_ip))
        if set_ip.get('tos'):
            cmd += ' tos'
            if set_ip['tos'].get('max_reliability'):
                cmd += ' max-reliability'
            elif set_ip['tos'].get('max_throughput'):
                cmd += ' max-throughput'
            elif set_ip['tos'].get('min_delay'):
                cmd += ' min-delay'
            elif set_ip['tos'].get('min_monetary_cost'):
                cmd += ' min-monetary-cost'
            elif set_ip['tos'].get('normal'):
                cmd += ' normal'
            command.append(cmd)
        if set_ip.get('vrf'):
            cmd += ' vrf {vrf} next-hop'.format(**set_ip)
            if set_ip['vrf'].get('verify_availability').get('address'):
                cmd += ' verify-availability {address} {sequence} track {track}'.format(**set_ip['vrf']['verify_availability'])
            elif set_ip['vrf'].get('address'):
                cmd += ' {address}'.format(**set_ip['vrf'])
            command.append(cmd)
        return command