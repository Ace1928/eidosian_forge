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
def _build_layout(self):
    self._activity_dot = HTML('●', css_classes=['activity-dot'], visible=self.param.show_activity_dot, stylesheets=self._stylesheets + self.param.stylesheets.rx())
    self._left_col = left_col = Column(self._render_avatar(), max_width=60, height=100, css_classes=['left'], stylesheets=self._stylesheets + self.param.stylesheets.rx(), visible=self.param.show_avatar, sizing_mode=None)
    self.param.watch(self._update_avatar_pane, 'avatar')
    self._object_panel = self._create_panel(self.object)
    self._update_chat_copy_icon()
    self._center_row = Row(self._object_panel, self.reaction_icons, css_classes=['center'], stylesheets=self._stylesheets + self.param.stylesheets.rx(), sizing_mode=None)
    self.param.watch(self._update_object_pane, 'object')
    self._user_html = HTML(self.param.user, height=20, css_classes=['name'], visible=self.param.show_user, stylesheets=self._stylesheets)
    header_row = Row(self._user_html, *self.param.header_objects.rx(), self.chat_copy_icon, self._activity_dot, stylesheets=self._stylesheets + self.param.stylesheets.rx(), sizing_mode='stretch_width', css_classes=['header'])
    self._timestamp_html = HTML(self.param.timestamp.rx().strftime(self.param.timestamp_format), css_classes=['timestamp'], visible=self.param.show_timestamp)
    footer_col = Column(*self.param.footer_objects.rx(), self._timestamp_html, stylesheets=self._stylesheets + self.param.stylesheets.rx(), sizing_mode='stretch_width', css_classes=['footer'])
    self._right_col = right_col = Column(header_row, self._center_row, footer_col, css_classes=['right'], stylesheets=self._stylesheets + self.param.stylesheets.rx(), sizing_mode=None)
    viewable_params = {p: self.param[p] for p in self.param if p in Viewable.param if p in Viewable.param and p != 'name'}
    viewable_params['stylesheets'] = self._stylesheets + self.param.stylesheets.rx()
    self._composite = Row(left_col, right_col, **viewable_params)