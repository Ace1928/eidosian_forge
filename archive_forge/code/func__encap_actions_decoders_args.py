import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _encap_actions_decoders_args():
    """Returns the decoders arguments for the encap actions."""
    return {'pop_vlan': decode_flag, 'strip_vlan': decode_flag, 'push_vlan': decode_default, 'pop_mpls': decode_int, 'push_mpls': decode_int, 'decap': decode_flag, 'encap': decode_encap}