from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.attributionReportingTriggerRegistered')
@dataclass
class AttributionReportingTriggerRegistered:
    """
    **EXPERIMENTAL**


    """
    registration: AttributionReportingTriggerRegistration
    event_level: AttributionReportingEventLevelResult
    aggregatable: AttributionReportingAggregatableResult

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttributionReportingTriggerRegistered:
        return cls(registration=AttributionReportingTriggerRegistration.from_json(json['registration']), event_level=AttributionReportingEventLevelResult.from_json(json['eventLevel']), aggregatable=AttributionReportingAggregatableResult.from_json(json['aggregatable']))