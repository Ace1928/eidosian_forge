from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_params(config_data):
    command = 'bgp'
    if config_data['bgp_params'].get('additional_paths'):
        command += ' additional-paths ' + config_data['bgp_params']['additional_paths']
        if config_data['bgp_params']['additional_paths'] == 'send':
            command += ' any'
    elif config_data['bgp_params'].get('advertise_inactive'):
        command += ' advertise-inactive'
    elif config_data['bgp_params'].get('allowas_in'):
        command += ' allowas-in'
        if config_data['bgp_params']['allowas_in'].get('count'):
            command += ' {count}'.format(**config_data['bgp_params']['allowas_in'])
    elif config_data['bgp_params'].get('always_compare_med'):
        command += ' always-comapre-med'
    elif config_data['bgp_params'].get('asn'):
        command += ' asn notaion {asn}'.format(**config_data['bgp_params'])
    elif config_data['bgp_params'].get('auto_local_addr'):
        command += ' auto-local-addr'
    elif config_data['bgp_params'].get('bestpath'):
        if config_data['bgp_params']['bestpath'].get('as_path'):
            command += ' bestpath as-path {as_path}'.format(**config_data['bgp_params']['bestpath'])
        elif config_data['bgp_params']['bestpath'].get('ecmp_fast'):
            command += ' bestpath ecmp-fast'
        elif config_data['bgp_params']['bestpath'].get('med'):
            command += ' bestpath med'
            if config_data['bgp_params']['bestpath']['med'].get('confed'):
                command += ' confed'
            else:
                command += ' missing-as-worst'
        elif config_data['bgp_params']['bestpath'].get('skip'):
            command += ' bestpath skip next-hop igp-cost'
        elif config_data['bgp_params']['bestpath'].get('tie_break'):
            tie = re.sub('_', '-', config_data['bgp_params']['bestpath']['tie_break'])
            command += ' tie-break ' + tie
    elif config_data['bgp_params'].get('client_to_client'):
        command += ' client-to-client'
    elif config_data['bgp_params'].get('cluster_id'):
        command += ' cluster-id {cluster_id}'.format(**config_data['bgp_params'])
    elif config_data['bgp_params'].get('confederation'):
        command += ' confederation'
        if config_data['bgp_params']['confederation'].get('identifier'):
            command += ' identifier ' + config_data['bgp_params']['confederation']['identifier']
        else:
            command += ' peers {peers}'.format(**config_data['bgp_params']['confederation'])
    elif config_data['bgp_params'].get('control_plane_filter'):
        command += ' control-plane-filter default-allow'
    elif config_data['bgp_params'].get('convergence'):
        command += ' convergence'
        if config_data['bgp_params']['convergence'].get('slow_peer'):
            command += ' slow-peer'
        command += ' time {time}'.format(**config_data['bgp_params']['convergence'])
    elif config_data['bgp_params'].get('default'):
        command += ' default {default}'.format(**config_data['bgp_params'])
    elif config_data['bgp_params'].get('enforce_first_as'):
        command += ' enforce-first-as'
    elif config_data['bgp_params'].get('host_routes'):
        command += ' host-routes fib direct-install'
    elif config_data['bgp_params'].get('labeled_unicast'):
        command += ' labeled-unicast rib {labeled_unicast}'.format(**config_data['bgp_params'])
    elif config_data['bgp_params'].get('listen'):
        command = 'dynamic peer max '
        if config_data['bgp_params']['listen'].get('limit'):
            command += '{limit}'.format(**config_data['bgp_params']['listen'])
        else:
            command += ' range {address} peer group'.format(**config_data['bgp_params']['listen']['range'])
            if config_data['bgp_params']['listen']['range']['peer_group'].get('peer_filter'):
                command += ' {name} peer-filter {peer_filter}'.format(**config_data['bgp_params']['listen']['range']['peer_group'])
            else:
                command += ' {name} remote-as {remote_as}'.format(**config_data['bgp_params']['listen']['range']['peer_group'])
    elif config_data['bgp_params'].get('log_neighbor_changes'):
        command += ' log-neighbor-changes'
    elif config_data['bgp_params'].get('missing_policy'):
        command += ' missing-policy direction {direction} action {action}'.format(**config_data['bgp_params']['missing_policy'])
    elif config_data['bgp_params'].get('monitoring'):
        command += ' monitoring'
    elif config_data['bgp_params'].get('next_hop_unchanged'):
        command += ' next-hop-unchanged'
    elif config_data['bgp_params'].get('redistribute_internal'):
        command += ' redistribute-internal'
    elif config_data['bgp_params'].get('route'):
        command += ' route install-map {route}'.format(**config_data['bgp_params'])
    elif config_data['bgp_params'].get('route_reflector'):
        command += ' route-reflector preserve-attributes'
        if config_data['bgp_params']['route_reflector'].get('preserve'):
            command += ' always'
    elif config_data['bgp_params'].get('transport'):
        command += ' transport listen-port {transport}'.format(**config_data['bgp_params'])
    return command