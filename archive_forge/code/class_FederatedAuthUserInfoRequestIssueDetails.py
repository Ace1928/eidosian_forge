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
class FederatedAuthUserInfoRequestIssueDetails:
    federated_auth_user_info_request_issue_reason: FederatedAuthUserInfoRequestIssueReason

    def to_json(self):
        json = dict()
        json['federatedAuthUserInfoRequestIssueReason'] = self.federated_auth_user_info_request_issue_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(federated_auth_user_info_request_issue_reason=FederatedAuthUserInfoRequestIssueReason.from_json(json['federatedAuthUserInfoRequestIssueReason']))