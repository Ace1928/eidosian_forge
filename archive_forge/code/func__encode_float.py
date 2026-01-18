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
def _encode_float(self, obj: float) -> NumberRep | float:
    if isnan(obj):
        return NumberRep(type='number', value='nan')
    elif isinf(obj):
        return NumberRep(type='number', value='-inf' if obj < 0 else '+inf')
    else:
        return obj