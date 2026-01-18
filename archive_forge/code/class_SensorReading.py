from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class SensorReading:
    single: typing.Optional[SensorReadingSingle] = None
    xyz: typing.Optional[SensorReadingXYZ] = None
    quaternion: typing.Optional[SensorReadingQuaternion] = None

    def to_json(self):
        json = dict()
        if self.single is not None:
            json['single'] = self.single.to_json()
        if self.xyz is not None:
            json['xyz'] = self.xyz.to_json()
        if self.quaternion is not None:
            json['quaternion'] = self.quaternion.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(single=SensorReadingSingle.from_json(json['single']) if 'single' in json else None, xyz=SensorReadingXYZ.from_json(json['xyz']) if 'xyz' in json else None, quaternion=SensorReadingQuaternion.from_json(json['quaternion']) if 'quaternion' in json else None)