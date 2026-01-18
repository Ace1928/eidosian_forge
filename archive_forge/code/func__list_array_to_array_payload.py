from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _list_array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    """Serialize list (regular and large) arrays to PicklableArrayPayload."""
    buffers = a.buffers()
    assert len(buffers) > 1, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    offset_buf = buffers[1]
    offset_buf, child_offset, child_length = _copy_offsets_buffer_if_needed(offset_buf, a.type, a.offset, len(a))
    child = a.values.slice(child_offset, child_length)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, offset_buf], null_count=a.null_count, offset=0, children=[_array_to_array_payload(child)])