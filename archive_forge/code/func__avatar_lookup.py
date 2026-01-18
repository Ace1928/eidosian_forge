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
def _avatar_lookup(self, user: str) -> Avatar:
    """
        Lookup the avatar for the user.
        """
    alpha_numeric_key = self._to_alpha_numeric(user)
    updated_avatars = DEFAULT_AVATARS.copy()
    updated_avatars.update(self.default_avatars)
    updated_avatars = {self._to_alpha_numeric(key): value for key, value in updated_avatars.items()}
    return updated_avatars.get(alpha_numeric_key, self.avatar).format(dist_path=CDN_DIST)