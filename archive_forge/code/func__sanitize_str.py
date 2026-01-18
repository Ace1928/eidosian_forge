import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def _sanitize_str(s: Union[str, bytes]) -> str:
    if isinstance(s, str):
        sanitized = s
    elif isinstance(s, bytes):
        sanitized = s.decode('utf-8', errors='ignore')
    else:
        sanitized = str(s)
    if len(sanitized) < 64:
        return sanitized
    return sanitized[:64] + f'...<+len={len(sanitized) - 64}>'