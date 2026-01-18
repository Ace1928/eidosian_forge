from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class FilledField:
    html_type: str
    id_: str
    name: str
    value: str
    autofill_type: str
    filling_strategy: FillingStrategy
    field_id: dom.BackendNodeId

    def to_json(self):
        json = dict()
        json['htmlType'] = self.html_type
        json['id'] = self.id_
        json['name'] = self.name
        json['value'] = self.value
        json['autofillType'] = self.autofill_type
        json['fillingStrategy'] = self.filling_strategy.to_json()
        json['fieldId'] = self.field_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(html_type=str(json['htmlType']), id_=str(json['id']), name=str(json['name']), value=str(json['value']), autofill_type=str(json['autofillType']), filling_strategy=FillingStrategy.from_json(json['fillingStrategy']), field_id=dom.BackendNodeId.from_json(json['fieldId']))