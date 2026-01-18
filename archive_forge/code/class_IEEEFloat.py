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
class IEEEFloat(CFloat):

    def __init__(self, default_value=Undefined, **kwargs):
        super().__init__(default_value=default_value, **kwargs)
        self.metadata.setdefault('to_json', _ieee_to_json)