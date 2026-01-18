from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _null_array_to_array_payload(a: 'pyarrow.NullArray') -> 'PicklableArrayPayload':
    """Serialize null array to PicklableArrayPayload."""
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[None], null_count=a.null_count, offset=0, children=[])