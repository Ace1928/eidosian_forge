from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.storageBucketDeleted')
@dataclass
class StorageBucketDeleted:
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> StorageBucketDeleted:
        return cls(bucket_id=str(json['bucketId']))