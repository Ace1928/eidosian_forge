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
def _to_np_int16(self, data: np.ndarray):
    dtype = data.dtype
    if dtype in (np.float32, np.float64):
        data = (data * 32768.0).astype(np.int16)
    return data