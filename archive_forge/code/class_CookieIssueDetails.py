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
class CookieIssueDetails:
    """
    This information is currently necessary, as the front-end has a difficult
    time finding a specific cookie. With this, we can convey specific error
    information without the cookie.
    """
    cookie_warning_reasons: typing.List[CookieWarningReason]
    cookie_exclusion_reasons: typing.List[CookieExclusionReason]
    operation: CookieOperation
    cookie: typing.Optional[AffectedCookie] = None
    raw_cookie_line: typing.Optional[str] = None
    site_for_cookies: typing.Optional[str] = None
    cookie_url: typing.Optional[str] = None
    request: typing.Optional[AffectedRequest] = None

    def to_json(self):
        json = dict()
        json['cookieWarningReasons'] = [i.to_json() for i in self.cookie_warning_reasons]
        json['cookieExclusionReasons'] = [i.to_json() for i in self.cookie_exclusion_reasons]
        json['operation'] = self.operation.to_json()
        if self.cookie is not None:
            json['cookie'] = self.cookie.to_json()
        if self.raw_cookie_line is not None:
            json['rawCookieLine'] = self.raw_cookie_line
        if self.site_for_cookies is not None:
            json['siteForCookies'] = self.site_for_cookies
        if self.cookie_url is not None:
            json['cookieUrl'] = self.cookie_url
        if self.request is not None:
            json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(cookie_warning_reasons=[CookieWarningReason.from_json(i) for i in json['cookieWarningReasons']], cookie_exclusion_reasons=[CookieExclusionReason.from_json(i) for i in json['cookieExclusionReasons']], operation=CookieOperation.from_json(json['operation']), cookie=AffectedCookie.from_json(json['cookie']) if 'cookie' in json else None, raw_cookie_line=str(json['rawCookieLine']) if 'rawCookieLine' in json else None, site_for_cookies=str(json['siteForCookies']) if 'siteForCookies' in json else None, cookie_url=str(json['cookieUrl']) if 'cookieUrl' in json else None, request=AffectedRequest.from_json(json['request']) if 'request' in json else None)