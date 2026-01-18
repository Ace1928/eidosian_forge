from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class ButtonClick(ModelEvent):
    """ Announce a button click event on a Bokeh button widget.

    """
    event_name = 'button_click'

    def __init__(self, model: AbstractButton | None) -> None:
        from .models.widgets import AbstractButton, ToggleButtonGroup
        if model is not None and (not isinstance(model, (AbstractButton, ToggleButtonGroup))):
            clsname = self.__class__.__name__
            raise ValueError(f'{clsname} event only applies to button and button group models')
        super().__init__(model=model)