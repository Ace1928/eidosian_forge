from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class AttributionReportingSourceRegistrationResult(enum.Enum):
    SUCCESS = 'success'
    INTERNAL_ERROR = 'internalError'
    INSUFFICIENT_SOURCE_CAPACITY = 'insufficientSourceCapacity'
    INSUFFICIENT_UNIQUE_DESTINATION_CAPACITY = 'insufficientUniqueDestinationCapacity'
    EXCESSIVE_REPORTING_ORIGINS = 'excessiveReportingOrigins'
    PROHIBITED_BY_BROWSER_POLICY = 'prohibitedByBrowserPolicy'
    SUCCESS_NOISED = 'successNoised'
    DESTINATION_REPORTING_LIMIT_REACHED = 'destinationReportingLimitReached'
    DESTINATION_GLOBAL_LIMIT_REACHED = 'destinationGlobalLimitReached'
    DESTINATION_BOTH_LIMITS_REACHED = 'destinationBothLimitsReached'
    REPORTING_ORIGINS_PER_SITE_LIMIT_REACHED = 'reportingOriginsPerSiteLimitReached'
    EXCEEDS_MAX_CHANNEL_CAPACITY = 'exceedsMaxChannelCapacity'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)