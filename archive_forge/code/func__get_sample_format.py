import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def _get_sample_format(dtype: torch.dtype) -> str:
    dtype_to_format = {torch.uint8: 'u8', torch.int16: 's16', torch.int32: 's32', torch.int64: 's64', torch.float32: 'flt', torch.float64: 'dbl'}
    format = dtype_to_format.get(dtype)
    if format is None:
        raise ValueError(f'No format found for dtype {dtype}; dtype must be one of {list(dtype_to_format.keys())}.')
    return format