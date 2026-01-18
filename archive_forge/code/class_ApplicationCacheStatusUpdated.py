from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
@event_class('ApplicationCache.applicationCacheStatusUpdated')
@dataclass
class ApplicationCacheStatusUpdated:
    frame_id: page.FrameId
    manifest_url: str
    status: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ApplicationCacheStatusUpdated:
        return cls(frame_id=page.FrameId.from_json(json['frameId']), manifest_url=str(json['manifestURL']), status=int(json['status']))