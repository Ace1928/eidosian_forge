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
class Face3(Tuple):
    """A trait for a named tuple corresponding to a three.js Face3.

    Accepts named tuples with the field names:
    ('a', 'b', 'c', 'normal', 'color', 'materialIndex')
    """
    klass = _castable_namedtuple('Face3', ('a', 'b', 'c', 'normal', 'color', 'materialIndex'))
    _cast_types = (list, tuple)
    info_text = 'a named tuple representing a Face3'

    def __init__(self, **kwargs):
        super(Face3, self).__init__(CInt(), CInt(), CInt(), Union([Vector3(allow_none=True), Tuple((Vector3(),) * 3)]), Union([Unicode(allow_none=True), Tuple((Unicode(),) * 3)]), CInt(allow_none=True), default_value=(0, 0, 0, None, None, None))
        self.metadata.setdefault('to_json', _face_to_json)