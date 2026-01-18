from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('DOMStorage.domStorageItemUpdated')
@dataclass
class DomStorageItemUpdated:
    storage_id: StorageId
    key: str
    old_value: str
    new_value: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DomStorageItemUpdated:
        return cls(storage_id=StorageId.from_json(json['storageId']), key=str(json['key']), old_value=str(json['oldValue']), new_value=str(json['newValue']))