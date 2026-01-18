from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class SensorReadingQuaternion:
    x: float
    y: float
    z: float
    w: float

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        json['z'] = self.z
        json['w'] = self.w
        return json

    @classmethod
    def from_json(cls, json):
        return cls(x=float(json['x']), y=float(json['y']), z=float(json['z']), w=float(json['w']))