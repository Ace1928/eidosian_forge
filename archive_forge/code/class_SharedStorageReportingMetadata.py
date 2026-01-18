from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class SharedStorageReportingMetadata:
    """
    Pair of reporting metadata details for a candidate URL for ``selectURL()``.
    """
    event_type: str
    reporting_url: str

    def to_json(self):
        json = dict()
        json['eventType'] = self.event_type
        json['reportingUrl'] = self.reporting_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(event_type=str(json['eventType']), reporting_url=str(json['reportingUrl']))