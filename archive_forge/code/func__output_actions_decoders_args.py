import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _output_actions_decoders_args():
    """Returns the decoder arguments for the output actions."""
    return {'output': decode_output, 'drop': decode_flag, 'controller': decode_controller, 'enqueue': nested_list_decoder(ListDecoders([('port', decode_default), ('queue', int)]), delims=[',', ':']), 'bundle': decode_bundle, 'bundle_load': decode_bundle_load, 'group': decode_default}