from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class PropertyPreview:
    name: str
    type_: str
    value: typing.Optional[str] = None
    value_preview: typing.Optional[ObjectPreview] = None
    subtype: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['type'] = self.type_
        if self.value is not None:
            json['value'] = self.value
        if self.value_preview is not None:
            json['valuePreview'] = self.value_preview.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), type_=str(json['type']), value=str(json['value']) if 'value' in json else None, value_preview=ObjectPreview.from_json(json['valuePreview']) if 'valuePreview' in json else None, subtype=str(json['subtype']) if 'subtype' in json else None)