from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.sharedStorageAccessed')
@dataclass
class SharedStorageAccessed:
    """
    Shared storage was accessed by the associated page.
    The following parameters are included in all events.
    """
    access_time: network.TimeSinceEpoch
    type_: SharedStorageAccessType
    main_frame_id: page.FrameId
    owner_origin: str
    params: SharedStorageAccessParams

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SharedStorageAccessed:
        return cls(access_time=network.TimeSinceEpoch.from_json(json['accessTime']), type_=SharedStorageAccessType.from_json(json['type']), main_frame_id=page.FrameId.from_json(json['mainFrameId']), owner_origin=str(json['ownerOrigin']), params=SharedStorageAccessParams.from_json(json['params']))