import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
@staticmethod
def _info_decoders_args():
    """Generate the decoder args for the info KVDecoders."""
    return {'packets': decode_int, 'bytes': decode_int, 'used': decode_time, 'flags': decode_default, 'dp': decode_default}