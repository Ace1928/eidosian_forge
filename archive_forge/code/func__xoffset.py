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
def _xoffset(self):
    min_xidx = [x0 for _, x0, _, _ in self.objects if x0 is not None]
    return min(min_xidx) if min_xidx and len(min_xidx) == len(self.objects) else 0