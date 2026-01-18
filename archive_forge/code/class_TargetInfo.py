from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@dataclass
class TargetInfo:
    target_id: TargetID
    type_: str
    title: str
    url: str
    attached: bool
    can_access_opener: bool
    opener_id: typing.Optional[TargetID] = None
    opener_frame_id: typing.Optional[page.FrameId] = None
    browser_context_id: typing.Optional[browser.BrowserContextID] = None
    subtype: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['targetId'] = self.target_id.to_json()
        json['type'] = self.type_
        json['title'] = self.title
        json['url'] = self.url
        json['attached'] = self.attached
        json['canAccessOpener'] = self.can_access_opener
        if self.opener_id is not None:
            json['openerId'] = self.opener_id.to_json()
        if self.opener_frame_id is not None:
            json['openerFrameId'] = self.opener_frame_id.to_json()
        if self.browser_context_id is not None:
            json['browserContextId'] = self.browser_context_id.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json):
        return cls(target_id=TargetID.from_json(json['targetId']), type_=str(json['type']), title=str(json['title']), url=str(json['url']), attached=bool(json['attached']), can_access_opener=bool(json['canAccessOpener']), opener_id=TargetID.from_json(json['openerId']) if 'openerId' in json else None, opener_frame_id=page.FrameId.from_json(json['openerFrameId']) if 'openerFrameId' in json else None, browser_context_id=browser.BrowserContextID.from_json(json['browserContextId']) if 'browserContextId' in json else None, subtype=str(json['subtype']) if 'subtype' in json else None)