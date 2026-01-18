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
def _update_avatar_pane(self, event=None):
    new_avatar = self._render_avatar()
    old_type = type(self._left_col[0])
    new_type = type(new_avatar)
    if isinstance(event.old, (HTML, ImageBase)) or new_type is not old_type:
        self._left_col[:] = [new_avatar]
    else:
        params = new_avatar.param.values()
        del params['name']
        self._left_col[0].param.update(**params)