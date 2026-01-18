import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
@staticmethod
def action_decoders():
    """Return the KVDecoders instance to parse the actions section.

        Uses the cached version if available.
        """
    if not ODPFlow._action_decoders:
        ODPFlow._action_decoders = ODPFlow._gen_action_decoders()
    return ODPFlow._action_decoders