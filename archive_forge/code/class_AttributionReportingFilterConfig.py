from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingFilterConfig:
    filter_values: typing.List[AttributionReportingFilterDataEntry]
    lookback_window: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['filterValues'] = [i.to_json() for i in self.filter_values]
        if self.lookback_window is not None:
            json['lookbackWindow'] = self.lookback_window
        return json

    @classmethod
    def from_json(cls, json):
        return cls(filter_values=[AttributionReportingFilterDataEntry.from_json(i) for i in json['filterValues']], lookback_window=int(json['lookbackWindow']) if 'lookbackWindow' in json else None)