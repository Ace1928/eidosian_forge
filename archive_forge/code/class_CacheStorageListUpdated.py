from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.cacheStorageListUpdated')
@dataclass
class CacheStorageListUpdated:
    """
    A cache has been added/deleted.
    """
    origin: str
    storage_key: str
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CacheStorageListUpdated:
        return cls(origin=str(json['origin']), storage_key=str(json['storageKey']), bucket_id=str(json['bucketId']))