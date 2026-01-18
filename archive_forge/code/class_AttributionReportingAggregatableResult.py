from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class AttributionReportingAggregatableResult(enum.Enum):
    SUCCESS = 'success'
    INTERNAL_ERROR = 'internalError'
    NO_CAPACITY_FOR_ATTRIBUTION_DESTINATION = 'noCapacityForAttributionDestination'
    NO_MATCHING_SOURCES = 'noMatchingSources'
    EXCESSIVE_ATTRIBUTIONS = 'excessiveAttributions'
    EXCESSIVE_REPORTING_ORIGINS = 'excessiveReportingOrigins'
    NO_HISTOGRAMS = 'noHistograms'
    INSUFFICIENT_BUDGET = 'insufficientBudget'
    NO_MATCHING_SOURCE_FILTER_DATA = 'noMatchingSourceFilterData'
    NOT_REGISTERED = 'notRegistered'
    PROHIBITED_BY_BROWSER_POLICY = 'prohibitedByBrowserPolicy'
    DEDUPLICATED = 'deduplicated'
    REPORT_WINDOW_PASSED = 'reportWindowPassed'
    EXCESSIVE_REPORTS = 'excessiveReports'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)