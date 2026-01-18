from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSKeyframeRule:
    """
    CSS keyframe rule representation.
    """
    origin: StyleSheetOrigin
    key_text: Value
    style: CSSStyle
    style_sheet_id: typing.Optional[StyleSheetId] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin.to_json()
        json['keyText'] = self.key_text.to_json()
        json['style'] = self.style.to_json()
        if self.style_sheet_id is not None:
            json['styleSheetId'] = self.style_sheet_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(origin=StyleSheetOrigin.from_json(json['origin']), key_text=Value.from_json(json['keyText']), style=CSSStyle.from_json(json['style']), style_sheet_id=StyleSheetId.from_json(json['styleSheetId']) if 'styleSheetId' in json else None)