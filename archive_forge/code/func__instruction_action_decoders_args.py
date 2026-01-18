import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _instruction_action_decoders_args():
    """Generate the decoder arguments for instruction actions
        (see man(7) ovs-actions)."""
    return {'meter': decode_int, 'clear_actions': decode_flag, 'write_metadata': decode_mask(64), 'goto_table': decode_int}