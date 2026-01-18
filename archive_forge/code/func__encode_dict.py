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
def _encode_dict(self, obj: dict[Any, Any]) -> MapRep:
    if len(obj) == 0:
        result = MapRep(type='map')
    else:
        result = MapRep(type='map', entries=[(self.encode(key), self.encode(val)) for key, val in obj.items()])
    return result