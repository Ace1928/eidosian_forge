import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _meta_action_decoders_args():
    """Returns the decoders arguments for the metadata actions."""
    meta_default_decoders = ['set_tunnel', 'set_tunnel64', 'set_queue']
    return {'pop_queue': decode_flag, **{field: decode_default for field in meta_default_decoders}}