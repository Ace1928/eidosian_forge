from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class SetCookieBlockedReason(enum.Enum):
    """
    Types of reasons why a cookie may not be stored from a response.
    """
    SECURE_ONLY = 'SecureOnly'
    SAME_SITE_STRICT = 'SameSiteStrict'
    SAME_SITE_LAX = 'SameSiteLax'
    SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = 'SameSiteUnspecifiedTreatedAsLax'
    SAME_SITE_NONE_INSECURE = 'SameSiteNoneInsecure'
    USER_PREFERENCES = 'UserPreferences'
    THIRD_PARTY_PHASEOUT = 'ThirdPartyPhaseout'
    THIRD_PARTY_BLOCKED_IN_FIRST_PARTY_SET = 'ThirdPartyBlockedInFirstPartySet'
    SYNTAX_ERROR = 'SyntaxError'
    SCHEME_NOT_SUPPORTED = 'SchemeNotSupported'
    OVERWRITE_SECURE = 'OverwriteSecure'
    INVALID_DOMAIN = 'InvalidDomain'
    INVALID_PREFIX = 'InvalidPrefix'
    UNKNOWN_ERROR = 'UnknownError'
    SCHEMEFUL_SAME_SITE_STRICT = 'SchemefulSameSiteStrict'
    SCHEMEFUL_SAME_SITE_LAX = 'SchemefulSameSiteLax'
    SCHEMEFUL_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = 'SchemefulSameSiteUnspecifiedTreatedAsLax'
    SAME_PARTY_FROM_CROSS_PARTY_CONTEXT = 'SamePartyFromCrossPartyContext'
    SAME_PARTY_CONFLICTS_WITH_OTHER_ATTRIBUTES = 'SamePartyConflictsWithOtherAttributes'
    NAME_VALUE_PAIR_EXCEEDS_MAX_SIZE = 'NameValuePairExceedsMaxSize'
    DISALLOWED_CHARACTER = 'DisallowedCharacter'
    NO_COOKIE_CONTENT = 'NoCookieContent'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)