from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSProperty:
    """
    CSS property declaration data.
    """
    name: str
    value: str
    important: typing.Optional[bool] = None
    implicit: typing.Optional[bool] = None
    text: typing.Optional[str] = None
    parsed_ok: typing.Optional[bool] = None
    disabled: typing.Optional[bool] = None
    range_: typing.Optional[SourceRange] = None
    longhand_properties: typing.Optional[typing.List[CSSProperty]] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        if self.important is not None:
            json['important'] = self.important
        if self.implicit is not None:
            json['implicit'] = self.implicit
        if self.text is not None:
            json['text'] = self.text
        if self.parsed_ok is not None:
            json['parsedOk'] = self.parsed_ok
        if self.disabled is not None:
            json['disabled'] = self.disabled
        if self.range_ is not None:
            json['range'] = self.range_.to_json()
        if self.longhand_properties is not None:
            json['longhandProperties'] = [i.to_json() for i in self.longhand_properties]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), value=str(json['value']), important=bool(json['important']) if 'important' in json else None, implicit=bool(json['implicit']) if 'implicit' in json else None, text=str(json['text']) if 'text' in json else None, parsed_ok=bool(json['parsedOk']) if 'parsedOk' in json else None, disabled=bool(json['disabled']) if 'disabled' in json else None, range_=SourceRange.from_json(json['range']) if 'range' in json else None, longhand_properties=[CSSProperty.from_json(i) for i in json['longhandProperties']] if 'longhandProperties' in json else None)