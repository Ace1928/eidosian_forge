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
class Euler(Tuple):
    """A trait for a set of Euler angles.

    Expressed as a tuple of tree floats (the angles), and the order as a string.
    See the three.js docs for futher details.
    """
    info_text = 'a set of Euler angles'
    default_value = (0, 0, 0, 'XYZ')
    _accepted_orders = ['XYZ', 'YZX', 'ZXY', 'XZY', 'YXZ', 'ZYX']

    def __init__(self, default_value=Undefined, **kwargs):
        if default_value is Undefined:
            default_value = self.default_value
        else:
            self.default_value = default_value
        super(Euler, self).__init__(IEEEFloat(), IEEEFloat(), IEEEFloat(), Enum(self._accepted_orders, self._accepted_orders[0]), default_value=default_value, **kwargs)
        self.metadata.setdefault('to_json', _euler_to_json)