from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
@property
def _prev_stage(self):
    return self.prev_selector.value