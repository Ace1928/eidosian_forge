from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class QuirksModeIssueDetails:
    """
    Details for issues about documents in Quirks Mode
    or Limited Quirks Mode that affects page layouting.
    """
    is_limited_quirks_mode: bool
    document_node_id: dom.BackendNodeId
    url: str
    frame_id: page.FrameId
    loader_id: network.LoaderId

    def to_json(self):
        json = dict()
        json['isLimitedQuirksMode'] = self.is_limited_quirks_mode
        json['documentNodeId'] = self.document_node_id.to_json()
        json['url'] = self.url
        json['frameId'] = self.frame_id.to_json()
        json['loaderId'] = self.loader_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(is_limited_quirks_mode=bool(json['isLimitedQuirksMode']), document_node_id=dom.BackendNodeId.from_json(json['documentNodeId']), url=str(json['url']), frame_id=page.FrameId.from_json(json['frameId']), loader_id=network.LoaderId.from_json(json['loaderId']))