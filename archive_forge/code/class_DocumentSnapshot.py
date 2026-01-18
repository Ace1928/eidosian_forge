from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class DocumentSnapshot:
    """
    Document snapshot.
    """
    document_url: StringIndex
    title: StringIndex
    base_url: StringIndex
    content_language: StringIndex
    encoding_name: StringIndex
    public_id: StringIndex
    system_id: StringIndex
    frame_id: StringIndex
    nodes: NodeTreeSnapshot
    layout: LayoutTreeSnapshot
    text_boxes: TextBoxSnapshot
    scroll_offset_x: typing.Optional[float] = None
    scroll_offset_y: typing.Optional[float] = None
    content_width: typing.Optional[float] = None
    content_height: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['documentURL'] = self.document_url.to_json()
        json['title'] = self.title.to_json()
        json['baseURL'] = self.base_url.to_json()
        json['contentLanguage'] = self.content_language.to_json()
        json['encodingName'] = self.encoding_name.to_json()
        json['publicId'] = self.public_id.to_json()
        json['systemId'] = self.system_id.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['nodes'] = self.nodes.to_json()
        json['layout'] = self.layout.to_json()
        json['textBoxes'] = self.text_boxes.to_json()
        if self.scroll_offset_x is not None:
            json['scrollOffsetX'] = self.scroll_offset_x
        if self.scroll_offset_y is not None:
            json['scrollOffsetY'] = self.scroll_offset_y
        if self.content_width is not None:
            json['contentWidth'] = self.content_width
        if self.content_height is not None:
            json['contentHeight'] = self.content_height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(document_url=StringIndex.from_json(json['documentURL']), title=StringIndex.from_json(json['title']), base_url=StringIndex.from_json(json['baseURL']), content_language=StringIndex.from_json(json['contentLanguage']), encoding_name=StringIndex.from_json(json['encodingName']), public_id=StringIndex.from_json(json['publicId']), system_id=StringIndex.from_json(json['systemId']), frame_id=StringIndex.from_json(json['frameId']), nodes=NodeTreeSnapshot.from_json(json['nodes']), layout=LayoutTreeSnapshot.from_json(json['layout']), text_boxes=TextBoxSnapshot.from_json(json['textBoxes']), scroll_offset_x=float(json['scrollOffsetX']) if 'scrollOffsetX' in json else None, scroll_offset_y=float(json['scrollOffsetY']) if 'scrollOffsetY' in json else None, content_width=float(json['contentWidth']) if 'contentWidth' in json else None, content_height=float(json['contentHeight']) if 'contentHeight' in json else None)