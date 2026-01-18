from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _dictionary_array_to_array_payload(a: 'pyarrow.DictionaryArray') -> 'PicklableArrayPayload':
    """Serialize dictionary arrays to PicklableArrayPayload."""
    indices_payload = _array_to_array_payload(a.indices)
    dictionary_payload = _array_to_array_payload(a.dictionary)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[], null_count=a.null_count, offset=0, children=[indices_payload, dictionary_payload])