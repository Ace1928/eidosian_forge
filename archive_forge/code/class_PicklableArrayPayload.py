from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
@dataclass
class PicklableArrayPayload:
    """Picklable array payload, holding data buffers and array metadata.

    This is a helper container for pickling and reconstructing nested Arrow Arrays while
    ensuring that the buffers that underly zero-copy slice views are properly truncated.
    """
    type: 'pyarrow.DataType'
    length: int
    buffers: List['pyarrow.Buffer']
    null_count: int
    offset: int
    children: List['PicklableArrayPayload']

    @classmethod
    def from_array(self, a: 'pyarrow.Array') -> 'PicklableArrayPayload':
        """Create a picklable array payload from an Arrow Array.

        This will recursively accumulate data buffer and metadata payloads that are
        ready for pickling; namely, the data buffers underlying zero-copy slice views
        will be properly truncated.
        """
        return _array_to_array_payload(a)

    def to_array(self) -> 'pyarrow.Array':
        """Reconstruct an Arrow Array from this picklable payload."""
        return _array_payload_to_array(self)