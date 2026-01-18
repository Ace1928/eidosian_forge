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
class Folium(HTML):
    """
    The Folium pane wraps Folium map components.
    """
    sizing_mode = param.ObjectSelector(default='stretch_width', objects=['fixed', 'stretch_width', 'stretch_height', 'stretch_both', 'scale_width', 'scale_height', 'scale_both', None])
    priority: ClassVar[float | bool | None] = 0.6

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        return getattr(obj, '__module__', '').startswith('folium.') and hasattr(obj, '_repr_html_')

    def _transform_object(self, obj: Any) -> Dict[str, Any]:
        text = '' if obj is None else obj
        if hasattr(text, '_repr_html_'):
            text = text._repr_html_().replace(FOLIUM_BEFORE, FOLIUM_AFTER)
        return dict(object=escape(text))