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
def _initialize_canvas(self, canvas):
    canvas._device_pixel_ratio = 2 if self.high_dpi else 1
    canvas._handle_message(None, {'type': 'initialized'}, None)