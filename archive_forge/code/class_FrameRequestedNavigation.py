from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@event_class('Page.frameRequestedNavigation')
@dataclass
class FrameRequestedNavigation:
    """
    **EXPERIMENTAL**

    Fired when a renderer-initiated navigation is requested.
    Navigation may still be cancelled after the event is issued.
    """
    frame_id: FrameId
    reason: ClientNavigationReason
    url: str
    disposition: ClientNavigationDisposition

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameRequestedNavigation:
        return cls(frame_id=FrameId.from_json(json['frameId']), reason=ClientNavigationReason.from_json(json['reason']), url=str(json['url']), disposition=ClientNavigationDisposition.from_json(json['disposition']))