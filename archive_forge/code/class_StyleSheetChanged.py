from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@event_class('CSS.styleSheetChanged')
@dataclass
class StyleSheetChanged:
    """
    Fired whenever a stylesheet is changed as a result of the client operation.
    """
    style_sheet_id: StyleSheetId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> StyleSheetChanged:
        return cls(style_sheet_id=StyleSheetId.from_json(json['styleSheetId']))