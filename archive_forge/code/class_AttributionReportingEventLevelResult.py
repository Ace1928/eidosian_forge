from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class AttributionReportingEventLevelResult(enum.Enum):
    SUCCESS = 'success'
    SUCCESS_DROPPED_LOWER_PRIORITY = 'successDroppedLowerPriority'
    INTERNAL_ERROR = 'internalError'
    NO_CAPACITY_FOR_ATTRIBUTION_DESTINATION = 'noCapacityForAttributionDestination'
    NO_MATCHING_SOURCES = 'noMatchingSources'
    DEDUPLICATED = 'deduplicated'
    EXCESSIVE_ATTRIBUTIONS = 'excessiveAttributions'
    PRIORITY_TOO_LOW = 'priorityTooLow'
    NEVER_ATTRIBUTED_SOURCE = 'neverAttributedSource'
    EXCESSIVE_REPORTING_ORIGINS = 'excessiveReportingOrigins'
    NO_MATCHING_SOURCE_FILTER_DATA = 'noMatchingSourceFilterData'
    PROHIBITED_BY_BROWSER_POLICY = 'prohibitedByBrowserPolicy'
    NO_MATCHING_CONFIGURATIONS = 'noMatchingConfigurations'
    EXCESSIVE_REPORTS = 'excessiveReports'
    FALSELY_ATTRIBUTED_SOURCE = 'falselyAttributedSource'
    REPORT_WINDOW_PASSED = 'reportWindowPassed'
    NOT_REGISTERED = 'notRegistered'
    REPORT_WINDOW_NOT_STARTED = 'reportWindowNotStarted'
    NO_MATCHING_TRIGGER_DATA = 'noMatchingTriggerData'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)