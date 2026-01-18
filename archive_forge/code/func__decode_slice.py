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
def _decode_slice(self, obj: SliceRep) -> slice:
    start = self._decode(obj['start'])
    stop = self._decode(obj['stop'])
    step = self._decode(obj['step'])
    return slice(start, stop, step)