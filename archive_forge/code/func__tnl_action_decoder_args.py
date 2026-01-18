import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
@staticmethod
def _tnl_action_decoder_args():
    """Generate the decoder arguments for the tunnel actions."""
    return {'tnl_push': nested_kv_decoder(KVDecoders({'tnl_port': decode_default, 'header': nested_kv_decoder(KVDecoders({'size': decode_int, 'type': decode_int, 'eth': nested_kv_decoder(KVDecoders({'src': EthMask, 'dst': EthMask, 'dl_type': decode_int})), 'ipv4': nested_kv_decoder(KVDecoders({'src': IPMask, 'dst': IPMask, 'proto': decode_int, 'tos': decode_int, 'ttl': decode_int, 'frag': decode_int})), 'ipv6': nested_kv_decoder(KVDecoders({'src': IPMask, 'dst': IPMask, 'label': decode_int, 'proto': decode_int, 'tclass': decode_int, 'hlimit': decode_int})), 'udp': nested_kv_decoder(KVDecoders({'src': decode_int, 'dst': decode_int, 'csum': Mask16})), 'vxlan': nested_kv_decoder(KVDecoders({'flags': decode_int, 'vni': decode_int})), 'geneve': nested_kv_decoder(KVDecoders({'oam': decode_flag, 'crit': decode_flag, 'vni': decode_int, 'options': partial(decode_geneve, False)})), 'gre': decode_tnl_gre, 'erspan': nested_kv_decoder(KVDecoders({'ver': decode_int, 'sid': decode_int, 'idx': decode_int, 'dir': decode_int, 'hwid': decode_int})), 'gtpu': nested_kv_decoder(KVDecoders({'flags': decode_int, 'msgtype': decode_int, 'teid': decode_int}))})), 'out_port': decode_default}))}