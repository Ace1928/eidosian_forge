from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
class AxisDescriptor(AbstractAxisDescriptor):
    """Simple container for the axis data.

    Add more localisations?

    .. code:: python

        a1 = AxisDescriptor()
        a1.minimum = 1
        a1.maximum = 1000
        a1.default = 400
        a1.name = "weight"
        a1.tag = "wght"
        a1.labelNames['fa-IR'] = "قطر"
        a1.labelNames['en'] = "Wéíght"
        a1.map = [(1.0, 10.0), (400.0, 66.0), (1000.0, 990.0)]
        a1.axisOrdering = 1
        a1.axisLabels = [
            AxisLabelDescriptor(name="Regular", userValue=400, elidable=True)
        ]
        doc.addAxis(a1)
    """
    _attrs = ['tag', 'name', 'maximum', 'minimum', 'default', 'map', 'axisOrdering', 'axisLabels']

    def __init__(self, *, tag=None, name=None, labelNames=None, minimum=None, default=None, maximum=None, hidden=False, map=None, axisOrdering=None, axisLabels=None):
        super().__init__(tag=tag, name=name, labelNames=labelNames, hidden=hidden, map=map, axisOrdering=axisOrdering, axisLabels=axisLabels)
        self.minimum = minimum
        'number. The minimum value for this axis in user space.\n\n        MutatorMath + varLib.\n        '
        self.maximum = maximum
        'number. The maximum value for this axis in user space.\n\n        MutatorMath + varLib.\n        '
        self.default = default
        'number. The default value for this axis, i.e. when a new location is\n        created, this is the value this axis will get in user space.\n\n        MutatorMath + varLib.\n        '

    def serialize(self):
        return dict(tag=self.tag, name=self.name, labelNames=self.labelNames, maximum=self.maximum, minimum=self.minimum, default=self.default, hidden=self.hidden, map=self.map, axisOrdering=self.axisOrdering, axisLabels=self.axisLabels)

    def map_forward(self, v):
        """Maps value from axis mapping's input (user) to output (design)."""
        from fontTools.varLib.models import piecewiseLinearMap
        if not self.map:
            return v
        return piecewiseLinearMap(v, {k: v for k, v in self.map})

    def map_backward(self, v):
        """Maps value from axis mapping's output (design) to input (user)."""
        from fontTools.varLib.models import piecewiseLinearMap
        if isinstance(v, tuple):
            v = v[0]
        if not self.map:
            return v
        return piecewiseLinearMap(v, {v: k for k, v in self.map})