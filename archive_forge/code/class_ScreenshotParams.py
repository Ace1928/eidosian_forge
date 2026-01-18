from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class ScreenshotParams:
    """
    Encoding options for a screenshot.
    """
    format_: typing.Optional[str] = None
    quality: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        if self.format_ is not None:
            json['format'] = self.format_
        if self.quality is not None:
            json['quality'] = self.quality
        return json

    @classmethod
    def from_json(cls, json):
        return cls(format_=str(json['format']) if 'format' in json else None, quality=int(json['quality']) if 'quality' in json else None)