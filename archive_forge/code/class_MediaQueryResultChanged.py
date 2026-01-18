from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@event_class('CSS.mediaQueryResultChanged')
@dataclass
class MediaQueryResultChanged:
    """
    Fires whenever a MediaQuery result changes (for example, after a browser window has been
    resized.) The current implementation considers only viewport-dependent media features.
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> MediaQueryResultChanged:
        return cls()