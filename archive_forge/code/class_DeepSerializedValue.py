from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class DeepSerializedValue:
    """
    Represents deep serialized value.
    """
    type_: str
    value: typing.Optional[typing.Any] = None
    object_id: typing.Optional[str] = None
    weak_local_object_reference: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.value is not None:
            json['value'] = self.value
        if self.object_id is not None:
            json['objectId'] = self.object_id
        if self.weak_local_object_reference is not None:
            json['weakLocalObjectReference'] = self.weak_local_object_reference
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), value=json['value'] if 'value' in json else None, object_id=str(json['objectId']) if 'objectId' in json else None, weak_local_object_reference=int(json['weakLocalObjectReference']) if 'weakLocalObjectReference' in json else None)