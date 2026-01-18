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
def _euler_to_json(value, owner):
    if value is None:
        return None
    return _ieee_tuple_to_json(value[:3], owner) + [value[3]]