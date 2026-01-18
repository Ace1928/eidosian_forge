from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_mpls_traffic_eng(config_data):
    if 'traffic_eng' in config_data['mpls']:
        command = 'mpls traffic-eng'
        if 'area' in config_data['mpls']['traffic_eng']:
            command += ' area {area}'.format(**config_data['mpls']['traffic_eng'])
        elif 'autoroute_exclude' in config_data['mpls']['traffic_eng']:
            command += ' autoroute-exclude prefix-list {autoroute_exclude}'.format(**config_data['mpls']['traffic_eng'])
        elif 'interface' in config_data['mpls']['traffic_eng']:
            command += ' interface {int_type}'.format(**config_data['mpls']['traffic_eng']['interface'])
            if 'area' in config_data['mpls']['traffic_eng']['interface']:
                command += ' area {area}'.format(**config_data['mpls']['traffic_eng']['interface'])
        elif 'mesh_group' in config_data['mpls']['traffic_eng']:
            command += ' mesh-group {id} {interface}'.format(**config_data['mpls']['traffic_eng']['mesh_group'])
            if 'area' in config_data['mpls']['traffic_eng']['mesh_group']:
                command += ' area {area}'.format(**config_data['mpls']['traffic_eng']['mesh_group'])
        elif 'multicast_intact' in config_data['mpls']['traffic_eng']:
            command += ' multicast-intact'
        elif 'router_id_interface' in config_data['mpls']['traffic_eng']:
            command += ' router-id {router_id_interface}'.format(**config_data['mpls']['traffic_eng'])
        return command