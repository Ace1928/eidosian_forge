from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _tensor_array_to_array_payload(a: 'ArrowTensorArray') -> 'PicklableArrayPayload':
    """Serialize tensor arrays to PicklableArrayPayload."""
    storage_payload = _array_to_array_payload(a.storage)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[], null_count=a.null_count, offset=0, children=[storage_payload])