import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def action_to_str(act):
    action_type = act.cls_action_type
    if action_type == ofproto_v1_2.OFPAT_OUTPUT:
        port = UTIL.ofp_port_to_user(act.port)
        buf = 'OUTPUT:' + str(port)
    elif action_type == ofproto_v1_2.OFPAT_COPY_TTL_OUT:
        buf = 'COPY_TTL_OUT'
    elif action_type == ofproto_v1_2.OFPAT_COPY_TTL_IN:
        buf = 'COPY_TTL_IN'
    elif action_type == ofproto_v1_2.OFPAT_SET_MPLS_TTL:
        buf = 'SET_MPLS_TTL:' + str(act.mpls_ttl)
    elif action_type == ofproto_v1_2.OFPAT_DEC_MPLS_TTL:
        buf = 'DEC_MPLS_TTL'
    elif action_type == ofproto_v1_2.OFPAT_PUSH_VLAN:
        buf = 'PUSH_VLAN:' + str(act.ethertype)
    elif action_type == ofproto_v1_2.OFPAT_POP_VLAN:
        buf = 'POP_VLAN'
    elif action_type == ofproto_v1_2.OFPAT_PUSH_MPLS:
        buf = 'PUSH_MPLS:' + str(act.ethertype)
    elif action_type == ofproto_v1_2.OFPAT_POP_MPLS:
        buf = 'POP_MPLS:' + str(act.ethertype)
    elif action_type == ofproto_v1_2.OFPAT_SET_QUEUE:
        queue_id = UTIL.ofp_queue_to_user(act.queue_id)
        buf = 'SET_QUEUE:' + str(queue_id)
    elif action_type == ofproto_v1_2.OFPAT_GROUP:
        group_id = UTIL.ofp_group_to_user(act.group_id)
        buf = 'GROUP:' + str(group_id)
    elif action_type == ofproto_v1_2.OFPAT_SET_NW_TTL:
        buf = 'SET_NW_TTL:' + str(act.nw_ttl)
    elif action_type == ofproto_v1_2.OFPAT_DEC_NW_TTL:
        buf = 'DEC_NW_TTL'
    elif action_type == ofproto_v1_2.OFPAT_SET_FIELD:
        buf = 'SET_FIELD: {%s:%s}' % (act.key, act.value)
    else:
        buf = 'UNKNOWN'
    return buf