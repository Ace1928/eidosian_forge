from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import (
from .buttons import ButtonLike
from .widget import Widget
@abstract
class ToggleButtonGroup(AbstractGroup, ButtonLike):
    """ Abstract base class for groups with items rendered as buttons.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    orientation = Enum('horizontal', 'vertical', help='\n    Orient the button group either horizontally (default) or vertically.\n    ')