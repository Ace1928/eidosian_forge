import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
@staticmethod
def _gen_info_decoders():
    """Generate the info KVDecoders."""
    return KVDecoders(ODPFlow._info_decoders_args())