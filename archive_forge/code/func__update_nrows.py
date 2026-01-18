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
@param.depends('nrows', watch=True)
def _update_nrows(self):
    if not self._updating:
        self._rows_fixed = bool(self.nrows)