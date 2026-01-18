from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_af_neighbor(config_data):
    afi = config_data['neighbors']['address_family']['afi'] + '-unicast'
    command = 'protocols bgp {as_number} '.format(**config_data)
    command += 'neighbor {neighbor_address} address-family '.format(**config_data['neighbors']) + afi
    config_data = config_data['neighbors']['address_family']
    if config_data.get('allowas_in'):
        command += ' allowas-in number {allowas_in}'.format(**config_data)
    elif config_data.get('as_override'):
        command += ' as-override'
    elif config_data.get('capability'):
        command += ' capability '
        if config_data['capability'].get('dynamic'):
            command += 'dynamic'
        elif config_data['capability'].get('orf'):
            command += ' prefix-list {orf}'.format(**config_data['capability'])
    elif config_data.get('default_originate'):
        command += ' default-originate route-map {default_originate}'.format(**config_data)
    elif config_data.get('maximum_prefix'):
        command += ' maximum-prefix {maximum_prefix}'.format(**config_data)
    elif config_data.get('nexthop_local'):
        command += ' nexthop-local'
    elif config_data.get('nexthop_self'):
        command += ' nexthop-self'
    elif config_data.get('peer_group'):
        command += ' peer-group {peer_group}'.format(**config_data)
    elif config_data.get('remote_private_as'):
        command += ' remote-private-as'
    elif config_data.get('route_reflector_client'):
        command += ' route-reflector-client'
    elif config_data.get('route_server_client'):
        command += ' route-server-client'
    elif config_data.get('soft_reconfiguration'):
        command += ' soft-reconfiguration inbound'
    elif config_data.get('unsuppress_map'):
        command += ' unsuppress-map {unsuppress_map}'.format(**config_data)
    elif config_data.get('weight'):
        command += ' weight {weight}'.format(**config_data)
    return command