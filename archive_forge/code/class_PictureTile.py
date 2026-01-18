from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
@dataclass
class PictureTile:
    """
    Serialized fragment of layer picture along with its offset within the layer.
    """
    x: float
    y: float
    picture: str

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        json['picture'] = self.picture
        return json

    @classmethod
    def from_json(cls, json):
        return cls(x=float(json['x']), y=float(json['y']), picture=str(json['picture']))