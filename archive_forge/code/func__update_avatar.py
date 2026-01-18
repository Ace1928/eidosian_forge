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
@param.depends('avatar_lookup', 'user', watch=True)
def _update_avatar(self):
    """
        Update the avatar based on the user name.

        We do not use on_init here because if avatar is set,
        we don't want to override the provided avatar.

        However, if the user is updated, we want to update the avatar.
        """
    if self.avatar_lookup:
        self.avatar = self.avatar_lookup(self.user)
    else:
        self.avatar = self._avatar_lookup(self.user)