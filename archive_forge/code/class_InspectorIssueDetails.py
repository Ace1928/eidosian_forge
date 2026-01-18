from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class InspectorIssueDetails:
    """
    This struct holds a list of optional fields with additional information
    specific to the kind of issue. When adding a new issue code, please also
    add a new optional field to this type.
    """
    same_site_cookie_issue_details: typing.Optional[SameSiteCookieIssueDetails] = None
    mixed_content_issue_details: typing.Optional[MixedContentIssueDetails] = None
    blocked_by_response_issue_details: typing.Optional[BlockedByResponseIssueDetails] = None
    heavy_ad_issue_details: typing.Optional[HeavyAdIssueDetails] = None

    def to_json(self):
        json = dict()
        if self.same_site_cookie_issue_details is not None:
            json['sameSiteCookieIssueDetails'] = self.same_site_cookie_issue_details.to_json()
        if self.mixed_content_issue_details is not None:
            json['mixedContentIssueDetails'] = self.mixed_content_issue_details.to_json()
        if self.blocked_by_response_issue_details is not None:
            json['blockedByResponseIssueDetails'] = self.blocked_by_response_issue_details.to_json()
        if self.heavy_ad_issue_details is not None:
            json['heavyAdIssueDetails'] = self.heavy_ad_issue_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(same_site_cookie_issue_details=SameSiteCookieIssueDetails.from_json(json['sameSiteCookieIssueDetails']) if 'sameSiteCookieIssueDetails' in json else None, mixed_content_issue_details=MixedContentIssueDetails.from_json(json['mixedContentIssueDetails']) if 'mixedContentIssueDetails' in json else None, blocked_by_response_issue_details=BlockedByResponseIssueDetails.from_json(json['blockedByResponseIssueDetails']) if 'blockedByResponseIssueDetails' in json else None, heavy_ad_issue_details=HeavyAdIssueDetails.from_json(json['heavyAdIssueDetails']) if 'heavyAdIssueDetails' in json else None)