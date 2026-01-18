from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
class InspectorIssueCode(enum.Enum):
    """
    A unique identifier for the type of issue. Each type may use one of the
    optional fields in InspectorIssueDetails to convey more specific
    information about the kind of issue.
    """
    SAME_SITE_COOKIE_ISSUE = 'SameSiteCookieIssue'
    MIXED_CONTENT_ISSUE = 'MixedContentIssue'
    BLOCKED_BY_RESPONSE_ISSUE = 'BlockedByResponseIssue'
    HEAVY_AD_ISSUE = 'HeavyAdIssue'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)