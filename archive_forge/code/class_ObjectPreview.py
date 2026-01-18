from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class ObjectPreview:
    """
    Object containing abbreviated remote object value.
    """
    type_: str
    overflow: bool
    properties: typing.List[PropertyPreview]
    subtype: typing.Optional[str] = None
    description: typing.Optional[str] = None
    entries: typing.Optional[typing.List[EntryPreview]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['overflow'] = self.overflow
        json['properties'] = [i.to_json() for i in self.properties]
        if self.subtype is not None:
            json['subtype'] = self.subtype
        if self.description is not None:
            json['description'] = self.description
        if self.entries is not None:
            json['entries'] = [i.to_json() for i in self.entries]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), overflow=bool(json['overflow']), properties=[PropertyPreview.from_json(i) for i in json['properties']], subtype=str(json['subtype']) if 'subtype' in json else None, description=str(json['description']) if 'description' in json else None, entries=[EntryPreview.from_json(i) for i in json['entries']] if 'entries' in json else None)