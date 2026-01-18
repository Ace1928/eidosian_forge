import functools
from ovs.flow.kv import KVParser, KVDecoders, nested_kv_decoder
from ovs.flow.ofp_fields import field_decoders
from ovs.flow.flow import Flow, Section
from ovs.flow.list import ListDecoders, nested_list_decoder
from ovs.flow.decoders import (
from ovs.flow.ofp_act import (
@staticmethod
def _fw_action_decoders_args():
    """Returns the decoders arguments for the firewalling actions."""
    return {'ct': nested_kv_decoder(KVDecoders({'commit': decode_flag, 'zone': decode_zone, 'table': decode_int, 'nat': decode_nat, 'force': decode_flag, 'exec': nested_kv_decoder(KVDecoders({**OFPFlow._encap_actions_decoders_args(), **OFPFlow._field_action_decoders_args(), **OFPFlow._meta_action_decoders_args()}), is_list=True), 'alg': decode_default})), 'ct_clear': decode_flag, 'fin_timeout': nested_kv_decoder(KVDecoders({'idle_timeout': decode_time, 'hard_timeout': decode_time}))}