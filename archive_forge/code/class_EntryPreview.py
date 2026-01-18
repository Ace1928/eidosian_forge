from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class EntryPreview:
    value: ObjectPreview
    key: typing.Optional[ObjectPreview] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        if self.key is not None:
            json['key'] = self.key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(value=ObjectPreview.from_json(json['value']), key=ObjectPreview.from_json(json['key']) if 'key' in json else None)