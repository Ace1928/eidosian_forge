from __future__ import annotations
import math
from collections import namedtuple
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import FlexBox as BkFlexBox, GridBox as BkGridBox
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from .base import (
@property
def _yoffset(self):
    min_yidx = [y0 for y0, x0, _, _ in self.objects if y0 is not None]
    return min(min_yidx) if min_yidx and len(min_yidx) == len(self.objects) else 0