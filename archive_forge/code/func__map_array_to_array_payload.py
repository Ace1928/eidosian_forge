from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _map_array_to_array_payload(a: 'pyarrow.MapArray') -> 'PicklableArrayPayload':
    """Serialize map arrays to PicklableArrayPayload."""
    import pyarrow as pa
    buffers = a.buffers()
    assert len(buffers) > 0, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    new_buffers = [bitmap_buf]
    offset_buf = buffers[1]
    offset_buf, data_offset, data_length = _copy_offsets_buffer_if_needed(offset_buf, a.type, a.offset, len(a))
    if isinstance(a, pa.lib.ListArray):
        new_buffers.append(offset_buf)
        children = [_array_to_array_payload(a.values.slice(data_offset, data_length))]
    else:
        buffers = a.buffers()
        assert len(buffers) > 2, len(buffers)
        offsets = pa.Array.from_buffers(pa.int32(), len(a) + 1, [bitmap_buf, offset_buf])
        keys = a.keys.slice(data_offset, data_length)
        items = a.items.slice(data_offset, data_length)
        children = [_array_to_array_payload(offsets), _array_to_array_payload(keys), _array_to_array_payload(items)]
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=new_buffers, null_count=a.null_count, offset=0, children=children)