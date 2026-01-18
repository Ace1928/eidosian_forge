from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@event_class('Audits.issueAdded')
@dataclass
class IssueAdded:
    issue: InspectorIssue

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IssueAdded:
        return cls(issue=InspectorIssue.from_json(json['issue']))