from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSMedia:
    """
    CSS media rule descriptor.
    """
    text: str
    source: str
    source_url: typing.Optional[str] = None
    range_: typing.Optional[SourceRange] = None
    style_sheet_id: typing.Optional[StyleSheetId] = None
    media_list: typing.Optional[typing.List[MediaQuery]] = None

    def to_json(self):
        json = dict()
        json['text'] = self.text
        json['source'] = self.source
        if self.source_url is not None:
            json['sourceURL'] = self.source_url
        if self.range_ is not None:
            json['range'] = self.range_.to_json()
        if self.style_sheet_id is not None:
            json['styleSheetId'] = self.style_sheet_id.to_json()
        if self.media_list is not None:
            json['mediaList'] = [i.to_json() for i in self.media_list]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(text=str(json['text']), source=str(json['source']), source_url=str(json['sourceURL']) if 'sourceURL' in json else None, range_=SourceRange.from_json(json['range']) if 'range' in json else None, style_sheet_id=StyleSheetId.from_json(json['styleSheetId']) if 'styleSheetId' in json else None, media_list=[MediaQuery.from_json(i) for i in json['mediaList']] if 'mediaList' in json else None)