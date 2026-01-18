from __future__ import annotations
import logging # isort:skip
import base64
import datetime as dt
import sys
from array import array as TypedArray
from math import isinf, isnan
from types import SimpleNamespace
from typing import (
import numpy as np
from ..util.dataclasses import (
from ..util.dependencies import uses_pandas
from ..util.serialization import (
from ..util.warnings import BokehUserWarning, warn
from .types import ID
def _encode_dataclass(self, obj: Any) -> ObjectRep:
    cls = type(obj)
    module = cls.__module__
    name = cls.__qualname__.replace('<locals>.', '')
    rep = ObjectRep(type='object', name=f'{module}.{name}')
    attributes = list(entries(obj))
    if attributes:
        rep['attributes'] = {key: self.encode(val) for key, val in attributes}
    return rep