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
@event_class('Network.reportingApiEndpointsChangedForOrigin')
@dataclass
class ReportingApiEndpointsChangedForOrigin:
    """
    **EXPERIMENTAL**


    """
    origin: str
    endpoints: typing.List[ReportingApiEndpoint]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiEndpointsChangedForOrigin:
        return cls(origin=str(json['origin']), endpoints=[ReportingApiEndpoint.from_json(i) for i in json['endpoints']])