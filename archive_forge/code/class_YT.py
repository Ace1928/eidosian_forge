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
class YT(HTML):
    """
    YT panes wrap plottable objects from the YT library.
    By default, the height and width are calculated by summing all
    contained plots, but can optionally be specified explicitly to
    provide additional space.
    """
    priority: ClassVar[float | bool | None] = 0.5

    @classmethod
    def applies(cls, obj: bool) -> float | bool | None:
        return getattr(obj, '__module__', '').startswith('yt.') and hasattr(obj, 'plots') and hasattr(obj, '_repr_html_')