from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import (
from .buttons import ButtonLike
from .widget import Widget
class RadioButtonGroup(ToggleButtonGroup):
    """ A group of radio boxes rendered as toggle buttons.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    active = Nullable(Int, help='\n    The index of the selected radio box, or ``None`` if nothing is\n    selected.\n    ')