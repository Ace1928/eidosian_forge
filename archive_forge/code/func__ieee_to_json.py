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
def _ieee_to_json(value, owner):
    if isinstance(value, numbers.Real):
        if math.isnan(value) or math.isinf(value):
            return repr(value)
    return value