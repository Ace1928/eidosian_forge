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
class AbstractAxisDescriptor(SimpleDescriptor):
    flavor = 'axis'

    def __init__(self, *, tag=None, name=None, labelNames=None, hidden=False, map=None, axisOrdering=None, axisLabels=None):
        self.tag = tag
        'string. Four letter tag for this axis. Some might be\n        registered at the `OpenType\n        specification <https://www.microsoft.com/typography/otspec/fvar.htm#VAT>`__.\n        Privately-defined axis tags must begin with an uppercase letter and\n        use only uppercase letters or digits.\n        '
        self.name = name
        'string. Name of the axis as it is used in the location dicts.\n\n        MutatorMath + varLib.\n        '
        self.labelNames = labelNames or {}
        'dict. When defining a non-registered axis, it will be\n        necessary to define user-facing readable names for the axis. Keyed by\n        xml:lang code. Values are required to be ``unicode`` strings, even if\n        they only contain ASCII characters.\n        '
        self.hidden = hidden
        'bool. Whether this axis should be hidden in user interfaces.\n        '
        self.map = map or []
        'list of input / output values that can describe a warp of user space\n        to design space coordinates. If no map values are present, it is assumed\n        user space is the same as design space, as in [(minimum, minimum),\n        (maximum, maximum)].\n\n        varLib.\n        '
        self.axisOrdering = axisOrdering
        'STAT table field ``axisOrdering``.\n\n        See: `OTSpec STAT Axis Record <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-records>`_\n\n        .. versionadded:: 5.0\n        '
        self.axisLabels: List[AxisLabelDescriptor] = axisLabels or []
        'STAT table entries for Axis Value Tables format 1, 2, 3.\n\n        See: `OTSpec STAT Axis Value Tables <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-value-tables>`_\n\n        .. versionadded:: 5.0\n        '