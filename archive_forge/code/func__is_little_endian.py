import os
import sys
from typing import Dict, Optional, Union
import numpy as np
from safetensors import deserialize, safe_open, serialize, serialize_file
def _is_little_endian(tensor: np.ndarray) -> bool:
    byteorder = tensor.dtype.byteorder
    if byteorder == '=':
        if sys.byteorder == 'little':
            return True
        else:
            return False
    elif byteorder == '|':
        return True
    elif byteorder == '<':
        return True
    elif byteorder == '>':
        return False
    raise ValueError(f'Unexpected byte order {byteorder}')