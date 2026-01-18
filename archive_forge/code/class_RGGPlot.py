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
class RGGPlot(PNG):
    """
    An RGGPlot pane renders an r2py-based ggplot2 figure to png
    and wraps the base64-encoded data in a bokeh Div model.
    """
    height = param.Integer(default=400)
    width = param.Integer(default=400)
    dpi = param.Integer(default=144, bounds=(1, None))
    _rerender_params = PNG._rerender_params + ['object', 'dpi', 'width', 'height']
    _rename: ClassVar[Mapping[str, str | None]] = {'dpi': None}

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        return type(obj).__name__ == 'GGPlot' and hasattr(obj, 'r_repr')

    def _data(self, obj):
        from rpy2 import robjects
        from rpy2.robjects.lib import grdevices
        with grdevices.render_to_bytesio(grdevices.png, type='cairo-png', width=self.width, height=self.height, res=self.dpi, antialias='subpixel') as b:
            robjects.r('print')(obj)
        return b.getvalue()