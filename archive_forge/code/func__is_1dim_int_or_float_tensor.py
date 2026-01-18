from __future__ import annotations
import os
from base64 import b64encode
from io import BytesIO
from typing import (
import numpy as np
import param
from ..models import Audio as _BkAudio, Video as _BkVideo
from ..util import isfile, isurl
from .base import ModelPane
def _is_1dim_int_or_float_tensor(obj: Any) -> bool:
    return isinstance(obj, TensorLike) and obj.dim() == 1 and (str(obj.dtype) in _VALID_TORCH_DTYPES_FOR_AUDIO)