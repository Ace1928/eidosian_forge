from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class CookieWarningReason(enum.Enum):
    WARN_SAME_SITE_UNSPECIFIED_CROSS_SITE_CONTEXT = 'WarnSameSiteUnspecifiedCrossSiteContext'
    WARN_SAME_SITE_NONE_INSECURE = 'WarnSameSiteNoneInsecure'
    WARN_SAME_SITE_UNSPECIFIED_LAX_ALLOW_UNSAFE = 'WarnSameSiteUnspecifiedLaxAllowUnsafe'
    WARN_SAME_SITE_STRICT_LAX_DOWNGRADE_STRICT = 'WarnSameSiteStrictLaxDowngradeStrict'
    WARN_SAME_SITE_STRICT_CROSS_DOWNGRADE_STRICT = 'WarnSameSiteStrictCrossDowngradeStrict'
    WARN_SAME_SITE_STRICT_CROSS_DOWNGRADE_LAX = 'WarnSameSiteStrictCrossDowngradeLax'
    WARN_SAME_SITE_LAX_CROSS_DOWNGRADE_STRICT = 'WarnSameSiteLaxCrossDowngradeStrict'
    WARN_SAME_SITE_LAX_CROSS_DOWNGRADE_LAX = 'WarnSameSiteLaxCrossDowngradeLax'
    WARN_ATTRIBUTE_VALUE_EXCEEDS_MAX_SIZE = 'WarnAttributeValueExceedsMaxSize'
    WARN_DOMAIN_NON_ASCII = 'WarnDomainNonASCII'
    WARN_THIRD_PARTY_PHASEOUT = 'WarnThirdPartyPhaseout'
    WARN_CROSS_SITE_REDIRECT_DOWNGRADE_CHANGES_INCLUSION = 'WarnCrossSiteRedirectDowngradeChangesInclusion'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)