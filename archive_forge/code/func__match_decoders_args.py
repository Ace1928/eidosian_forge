import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
@staticmethod
def _match_decoders_args():
    """Generate the arguments for the match KVDecoders."""
    return {**ODPFlow._field_decoders_args(), 'encap': nested_kv_decoder(KVDecoders(ODPFlow._field_decoders_args()))}