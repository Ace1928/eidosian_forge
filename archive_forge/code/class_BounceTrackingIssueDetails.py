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
class BounceTrackingIssueDetails:
    """
    This issue warns about sites in the redirect chain of a finished navigation
    that may be flagged as trackers and have their state cleared if they don't
    receive a user interaction. Note that in this context 'site' means eTLD+1.
    For example, if the URL ``https://example.test:80/bounce`` was in the
    redirect chain, the site reported would be ``example.test``.
    """
    tracking_sites: typing.List[str]

    def to_json(self):
        json = dict()
        json['trackingSites'] = [i for i in self.tracking_sites]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(tracking_sites=[str(i) for i in json['trackingSites']])