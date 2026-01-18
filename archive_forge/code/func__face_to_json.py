from collections import namedtuple
from collections.abc import Sequence
import numbers
import math
import re
import warnings
from traitlets import (
from ipywidgets import widget_serialization
from ipydatawidgets import DataUnion, NDArrayWidget, shape_constraints
import numpy as np
def _face_to_json(value, owner):
    if value is None:
        return None
    value = list(value)
    if value[3] is not None:
        normal = list(value[3])
        for i, v in enumerate(normal):
            if isinstance(v, tuple):
                normal[i] = _ieee_tuple_to_json(v, owner)
            else:
                normal[i] = _ieee_to_json(v, owner)
        value[3] = normal
    return value