from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class MediaQueryExpression:
    """
    Media query expression descriptor.
    """
    value: float
    unit: str
    feature: str
    value_range: typing.Optional[SourceRange] = None
    computed_length: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value
        json['unit'] = self.unit
        json['feature'] = self.feature
        if self.value_range is not None:
            json['valueRange'] = self.value_range.to_json()
        if self.computed_length is not None:
            json['computedLength'] = self.computed_length
        return json

    @classmethod
    def from_json(cls, json):
        return cls(value=float(json['value']), unit=str(json['unit']), feature=str(json['feature']), value_range=SourceRange.from_json(json['valueRange']) if 'valueRange' in json else None, computed_length=float(json['computedLength']) if 'computedLength' in json else None)