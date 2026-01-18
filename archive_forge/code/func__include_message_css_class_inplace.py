from __future__ import annotations
import datetime
import re
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from tempfile import NamedTemporaryFile
from textwrap import indent
from typing import (
from zoneinfo import ZoneInfo
import param
from ..io.resources import CDN_DIST, get_dist_path
from ..io.state import state
from ..layout import Column, Row
from ..pane.base import PaneBase, ReplacementPane, panel as _panel
from ..pane.image import (
from ..pane.markup import (
from ..pane.media import Audio, Video
from ..param import ParamFunction
from ..viewable import Viewable
from ..widgets.base import Widget
from .icon import ChatCopyIcon, ChatReactionIcons
def _include_message_css_class_inplace(self, obj):
    if hasattr(obj, 'objects'):
        obj.objects[:] = [self._include_message_css_class_inplace(o) for o in obj.objects]
    else:
        obj = _panel(obj)
    is_markup = isinstance(obj, HTMLBasePane) and (not isinstance(obj, FileBase))
    if obj.css_classes or not is_markup:
        return obj
    if len(str(obj.object)) > 0:
        obj.css_classes = [*(css for css in obj.css_classes if css != 'message'), 'message']
    obj.sizing_mode = None
    return obj