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
class GenericIssueDetails:
    """
    Depending on the concrete errorType, different properties are set.
    """
    error_type: GenericIssueErrorType
    frame_id: typing.Optional[page.FrameId] = None
    violating_node_id: typing.Optional[dom.BackendNodeId] = None
    violating_node_attribute: typing.Optional[str] = None
    request: typing.Optional[AffectedRequest] = None

    def to_json(self):
        json = dict()
        json['errorType'] = self.error_type.to_json()
        if self.frame_id is not None:
            json['frameId'] = self.frame_id.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        if self.violating_node_attribute is not None:
            json['violatingNodeAttribute'] = self.violating_node_attribute
        if self.request is not None:
            json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(error_type=GenericIssueErrorType.from_json(json['errorType']), frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None, violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None, violating_node_attribute=str(json['violatingNodeAttribute']) if 'violatingNodeAttribute' in json else None, request=AffectedRequest.from_json(json['request']) if 'request' in json else None)