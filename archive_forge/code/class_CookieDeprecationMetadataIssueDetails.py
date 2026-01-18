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
class CookieDeprecationMetadataIssueDetails:
    """
    This issue warns about third-party sites that are accessing cookies on the
    current page, and have been permitted due to having a global metadata grant.
    Note that in this context 'site' means eTLD+1. For example, if the URL
    ``https://example.test:80/web_page`` was accessing cookies, the site reported
    would be ``example.test``.
    """
    allowed_sites: typing.List[str]

    def to_json(self):
        json = dict()
        json['allowedSites'] = [i for i in self.allowed_sites]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(allowed_sites=[str(i) for i in json['allowedSites']])