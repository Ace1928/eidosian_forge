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
@event_class('Network.reportingApiReportUpdated')
@dataclass
class ReportingApiReportUpdated:
    """
    **EXPERIMENTAL**


    """
    report: ReportingApiReport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiReportUpdated:
        return cls(report=ReportingApiReport.from_json(json['report']))