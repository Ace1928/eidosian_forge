from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class LargestContentfulPaint:
    """
    See https://github.com/WICG/LargestContentfulPaint and largest_contentful_paint.idl
    """
    render_time: network.TimeSinceEpoch
    load_time: network.TimeSinceEpoch
    size: float
    element_id: typing.Optional[str] = None
    url: typing.Optional[str] = None
    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['renderTime'] = self.render_time.to_json()
        json['loadTime'] = self.load_time.to_json()
        json['size'] = self.size
        if self.element_id is not None:
            json['elementId'] = self.element_id
        if self.url is not None:
            json['url'] = self.url
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(render_time=network.TimeSinceEpoch.from_json(json['renderTime']), load_time=network.TimeSinceEpoch.from_json(json['loadTime']), size=float(json['size']), element_id=str(json['elementId']) if 'elementId' in json else None, url=str(json['url']) if 'url' in json else None, node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None)