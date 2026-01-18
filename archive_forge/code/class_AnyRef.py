from __future__ import annotations
import logging # isort:skip
import typing
from .bases import Init, Property
class AnyRef(Any):
    """ Accept all values and force reference discovery. """

    @property
    def has_ref(self) -> bool:
        return True