from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.attributionReportingSourceRegistered')
@dataclass
class AttributionReportingSourceRegistered:
    """
    **EXPERIMENTAL**


    """
    registration: AttributionReportingSourceRegistration
    result: AttributionReportingSourceRegistrationResult

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttributionReportingSourceRegistered:
        return cls(registration=AttributionReportingSourceRegistration.from_json(json['registration']), result=AttributionReportingSourceRegistrationResult.from_json(json['result']))