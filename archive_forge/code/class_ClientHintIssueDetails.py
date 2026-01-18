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
class ClientHintIssueDetails:
    """
    This issue tracks client hints related issues. It's used to deprecate old
    features, encourage the use of new ones, and provide general guidance.
    """
    source_code_location: SourceCodeLocation
    client_hint_issue_reason: ClientHintIssueReason

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['clientHintIssueReason'] = self.client_hint_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']), client_hint_issue_reason=ClientHintIssueReason.from_json(json['clientHintIssueReason']))