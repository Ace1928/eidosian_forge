from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@event_class('CSS.fontsUpdated')
@dataclass
class FontsUpdated:
    """
    Fires whenever a web font is updated.  A non-empty font parameter indicates a successfully loaded
    web font.
    """
    font: typing.Optional[FontFace]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FontsUpdated:
        return cls(font=FontFace.from_json(json['font']) if 'font' in json else None)