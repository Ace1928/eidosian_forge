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
class FederatedAuthRequestIssueDetails:
    federated_auth_request_issue_reason: FederatedAuthRequestIssueReason

    def to_json(self):
        json = dict()
        json['federatedAuthRequestIssueReason'] = self.federated_auth_request_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(federated_auth_request_issue_reason=FederatedAuthRequestIssueReason.from_json(json['federatedAuthRequestIssueReason']))