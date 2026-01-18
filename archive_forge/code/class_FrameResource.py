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
@dataclass
class FrameResource:
    """
    Information about the Resource on the page.
    """
    url: str
    type_: network.ResourceType
    mime_type: str
    last_modified: typing.Optional[network.TimeSinceEpoch] = None
    content_size: typing.Optional[float] = None
    failed: typing.Optional[bool] = None
    canceled: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['type'] = self.type_.to_json()
        json['mimeType'] = self.mime_type
        if self.last_modified is not None:
            json['lastModified'] = self.last_modified.to_json()
        if self.content_size is not None:
            json['contentSize'] = self.content_size
        if self.failed is not None:
            json['failed'] = self.failed
        if self.canceled is not None:
            json['canceled'] = self.canceled
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), type_=network.ResourceType.from_json(json['type']), mime_type=str(json['mimeType']), last_modified=network.TimeSinceEpoch.from_json(json['lastModified']) if 'lastModified' in json else None, content_size=float(json['contentSize']) if 'contentSize' in json else None, failed=bool(json['failed']) if 'failed' in json else None, canceled=bool(json['canceled']) if 'canceled' in json else None)