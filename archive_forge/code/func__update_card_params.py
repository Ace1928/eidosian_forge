from __future__ import annotations
import asyncio
import traceback
from enum import Enum
from inspect import (
from io import BytesIO
from typing import (
import param
from .._param import Margin
from ..io.resources import CDN_DIST
from ..layout import Feed, ListPanel
from ..layout.card import Card
from ..layout.spacer import VSpacer
from ..pane.image import SVG
from .message import ChatMessage
@param.depends('card_params', watch=True)
def _update_card_params(self):
    card_params = self.card_params.copy()
    card_params.pop('stylesheets', None)
    self._card.param.update(**card_params)