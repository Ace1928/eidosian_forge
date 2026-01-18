from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_match_ipv6(config_data):
    if config_data.get('match') and config_data['match'].get('ipv6'):
        cmd = 'match ipv6'
        if config_data['match']['ipv6'].get('address'):
            cmd += ' address'
            if config_data['match']['ipv6']['address'].get('prefix_list'):
                cmd += ' prefix-list {prefix_list}'.format(**config_data['match']['ipv6']['address'])
            elif config_data['match']['ipv6']['address'].get('acl'):
                cmd += ' {acl}'.format(**config_data['match']['ipv6']['address'])
        if config_data['match']['ipv6'].get('flowspec'):
            cmd += ' flowspec'
            if config_data['match']['ipv6']['flowspec'].get('dest_pfx'):
                cmd += ' dest-pfx'
            elif config_data['match']['ipv6']['flowspec'].get('src_pfx'):
                cmd += ' src-pfx'
            if config_data['match']['ipv6']['flowspec'].get('prefix_list'):
                cmd += ' prefix-list {prefix_list}'.format(**config_data['match']['ipv6']['flowspec'])
            elif config_data['match']['ipv6']['flowspec'].get('acl'):
                cmd += ' {acl}'.format(**config_data['match']['ipv6']['flowspec'])
        if config_data['match']['ipv6'].get('next_hop'):
            cmd += ' next-hop'
            if config_data['match']['ipv6']['next_hop'].get('prefix_list'):
                cmd += ' prefix-list {prefix_list}'.format(**config_data['match']['ipv6']['next_hop'])
            elif config_data['match']['ipv6']['next_hop'].get('acl'):
                cmd += ' {acl}'.format(**config_data['match']['ipv6']['next_hop'])
        if config_data['match']['ipv6'].get('route_source'):
            cmd += ' route-source'
            if config_data['match']['ipv6']['route_source'].get('prefix_list'):
                cmd += ' prefix-list {prefix_list}'.format(**config_data['match']['ipv6']['route_source'])
            elif config_data['match']['ipv6']['route_source'].get('acl'):
                cmd += ' {acl}'.format(**config_data['match']['ipv6']['route_source'])
        return cmd