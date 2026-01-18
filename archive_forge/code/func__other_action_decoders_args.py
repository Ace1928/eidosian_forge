import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _other_action_decoders_args():
    """Generate the decoder arguments for other actions
        (see man(7) ovs-actions)."""
    return {'conjunction': nested_list_decoder(ListDecoders([('id', decode_int), ('k', decode_int), ('n', decode_int)]), delims=[',', '/']), 'note': decode_default, 'sample': nested_kv_decoder(KVDecoders({'probability': decode_int, 'collector_set_id': decode_int, 'obs_domain_id': decode_int, 'obs_point_id': decode_int, 'sampling_port': decode_default, 'ingress': decode_flag, 'egress': decode_flag}))}