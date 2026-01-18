from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
def _buffer_to_array(buffer: bytes, dtype: Any=np.float32) -> List[float]:
    return np.frombuffer(buffer, dtype=dtype).tolist()