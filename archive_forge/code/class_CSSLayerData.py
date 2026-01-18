from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSLayerData:
    """
    CSS Layer data.
    """
    name: str
    order: float
    sub_layers: typing.Optional[typing.List[CSSLayerData]] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['order'] = self.order
        if self.sub_layers is not None:
            json['subLayers'] = [i.to_json() for i in self.sub_layers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), order=float(json['order']), sub_layers=[CSSLayerData.from_json(i) for i in json['subLayers']] if 'subLayers' in json else None)