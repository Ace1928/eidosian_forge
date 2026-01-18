from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class CustomPreview:
    header: str
    body_getter_id: typing.Optional[RemoteObjectId] = None

    def to_json(self):
        json = dict()
        json['header'] = self.header
        if self.body_getter_id is not None:
            json['bodyGetterId'] = self.body_getter_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(header=str(json['header']), body_getter_id=RemoteObjectId.from_json(json['bodyGetterId']) if 'bodyGetterId' in json else None)