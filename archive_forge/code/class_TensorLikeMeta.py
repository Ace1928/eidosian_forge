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
class TensorLikeMeta(type):
    """See https://blog.finxter.com/python-__instancecheck__-magic-method/"""

    def __instancecheck__(self, instance):
        numpy_attr = getattr(instance, 'numpy', '')
        dim_attr = getattr(instance, 'dim', '')
        return bool(numpy_attr) and callable(numpy_attr) and callable(dim_attr) and hasattr(instance, 'dtype')