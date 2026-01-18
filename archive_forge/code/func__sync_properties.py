from __future__ import annotations
import re
import sys
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from typing import (
import param
from bokeh.models import (
from bokeh.themes import Theme
from ..io import remove_root, state
from ..io.notebook import push
from ..util import escape
from ..viewable import Layoutable
from .base import PaneBase
from .image import (
from .ipywidget import IPyWidget
from .markup import HTML
@param.depends('object', watch=True)
def _sync_properties(self):
    if self.object is None:
        return
    self._syncing_props = True
    try:
        self.param.update({p: v for p, v in self.object.properties_with_values().items() if p not in self._overrides and p in Layoutable.param and (p not in ('css_classes', 'name'))})
        props = {o: getattr(self, o) for o in self._overrides}
        if props:
            self.object.update(**props)
    finally:
        self._syncing_props = False