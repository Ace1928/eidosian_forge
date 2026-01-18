from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class HingeConfig:
    """
    Configuration for dual screen hinge
    """
    rect: dom.Rect
    content_color: typing.Optional[dom.RGBA] = None
    outline_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        json['rect'] = self.rect.to_json()
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.outline_color is not None:
            json['outlineColor'] = self.outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(rect=dom.Rect.from_json(json['rect']), content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None, outline_color=dom.RGBA.from_json(json['outlineColor']) if 'outlineColor' in json else None)