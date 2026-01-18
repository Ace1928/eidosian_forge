from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class OriginTrialToken:
    origin: str
    match_sub_domains: bool
    trial_name: str
    expiry_time: network.TimeSinceEpoch
    is_third_party: bool
    usage_restriction: OriginTrialUsageRestriction

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['matchSubDomains'] = self.match_sub_domains
        json['trialName'] = self.trial_name
        json['expiryTime'] = self.expiry_time.to_json()
        json['isThirdParty'] = self.is_third_party
        json['usageRestriction'] = self.usage_restriction.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(origin=str(json['origin']), match_sub_domains=bool(json['matchSubDomains']), trial_name=str(json['trialName']), expiry_time=network.TimeSinceEpoch.from_json(json['expiryTime']), is_third_party=bool(json['isThirdParty']), usage_restriction=OriginTrialUsageRestriction.from_json(json['usageRestriction']))