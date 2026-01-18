from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class CookieExclusionReason(enum.Enum):
    EXCLUDE_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = 'ExcludeSameSiteUnspecifiedTreatedAsLax'
    EXCLUDE_SAME_SITE_NONE_INSECURE = 'ExcludeSameSiteNoneInsecure'
    EXCLUDE_SAME_SITE_LAX = 'ExcludeSameSiteLax'
    EXCLUDE_SAME_SITE_STRICT = 'ExcludeSameSiteStrict'
    EXCLUDE_INVALID_SAME_PARTY = 'ExcludeInvalidSameParty'
    EXCLUDE_SAME_PARTY_CROSS_PARTY_CONTEXT = 'ExcludeSamePartyCrossPartyContext'
    EXCLUDE_DOMAIN_NON_ASCII = 'ExcludeDomainNonASCII'
    EXCLUDE_THIRD_PARTY_COOKIE_BLOCKED_IN_FIRST_PARTY_SET = 'ExcludeThirdPartyCookieBlockedInFirstPartySet'
    EXCLUDE_THIRD_PARTY_PHASEOUT = 'ExcludeThirdPartyPhaseout'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)