from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING
from .bases import Init
from .either import Either
from .instance import Instance
from .singletons import Intrinsic
from .string import MathString
class TextLike(Either):
    """ Accept a string that may be interpreted into text models or the models themselves.

    """

    def __init__(self, default: Init[str | BaseText]=Intrinsic, help: str | None=None) -> None:
        super().__init__(MathString(), Instance('bokeh.models.text.BaseText'), default=default, help=help)