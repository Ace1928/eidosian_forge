from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('DOMStorage.domStorageItemRemoved')
@dataclass
class DomStorageItemRemoved:
    storage_id: StorageId
    key: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DomStorageItemRemoved:
        return cls(storage_id=StorageId.from_json(json['storageId']), key=str(json['key']))