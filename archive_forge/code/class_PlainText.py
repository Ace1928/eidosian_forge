from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class PlainText(BaseText):
    """ Represents plain text in contexts where text parsing is allowed.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)