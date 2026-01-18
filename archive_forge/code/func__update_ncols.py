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
@param.depends('ncols', watch=True)
def _update_ncols(self):
    if not self._updating:
        self._cols_fixed = self.ncols is not None