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
def _make_matplotlib_svg_responsive(input_str):
    output_str = _width_regex.sub(b'width="100%"', input_str)
    output_str = _height_regex.sub(b'height="100%"', output_str)
    return output_str