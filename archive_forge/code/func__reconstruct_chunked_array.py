from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _reconstruct_chunked_array(chunks: List['PicklableArrayPayload'], type_: 'pyarrow.DataType') -> 'pyarrow.ChunkedArray':
    """Restore a serialized Arrow ChunkedArray from chunks and type."""
    import pyarrow as pa
    chunks = [chunk.to_array() for chunk in chunks]
    return pa.chunked_array(chunks, type_)