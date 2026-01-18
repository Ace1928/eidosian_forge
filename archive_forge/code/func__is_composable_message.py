from __future__ import annotations
from typing import Any
from streamlit.proto.Delta_pb2 import Delta
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
def _is_composable_message(msg: ForwardMsg) -> bool:
    """True if the ForwardMsg is potentially composable with other ForwardMsgs."""
    if not msg.HasField('delta'):
        return False
    delta_type = msg.delta.WhichOneof('type')
    return delta_type != 'add_rows' and delta_type != 'arrow_add_rows'