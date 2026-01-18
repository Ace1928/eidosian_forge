from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class SensorMetadata:
    available: typing.Optional[bool] = None
    minimum_frequency: typing.Optional[float] = None
    maximum_frequency: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        if self.available is not None:
            json['available'] = self.available
        if self.minimum_frequency is not None:
            json['minimumFrequency'] = self.minimum_frequency
        if self.maximum_frequency is not None:
            json['maximumFrequency'] = self.maximum_frequency
        return json

    @classmethod
    def from_json(cls, json):
        return cls(available=bool(json['available']) if 'available' in json else None, minimum_frequency=float(json['minimumFrequency']) if 'minimumFrequency' in json else None, maximum_frequency=float(json['maximumFrequency']) if 'maximumFrequency' in json else None)