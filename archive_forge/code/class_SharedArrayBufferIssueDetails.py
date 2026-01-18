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
class SharedArrayBufferIssueDetails:
    """
    Details for a issue arising from an SAB being instantiated in, or
    transferred to a context that is not cross-origin isolated.
    """
    source_code_location: SourceCodeLocation
    is_warning: bool
    type_: SharedArrayBufferIssueType

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['isWarning'] = self.is_warning
        json['type'] = self.type_.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']), is_warning=bool(json['isWarning']), type_=SharedArrayBufferIssueType.from_json(json['type']))