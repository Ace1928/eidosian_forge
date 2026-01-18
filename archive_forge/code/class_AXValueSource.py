from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class AXValueSource:
    """
    A single source for a computed AX property.
    """
    type_: AXValueSourceType
    value: typing.Optional[AXValue] = None
    attribute: typing.Optional[str] = None
    attribute_value: typing.Optional[AXValue] = None
    superseded: typing.Optional[bool] = None
    native_source: typing.Optional[AXValueNativeSourceType] = None
    native_source_value: typing.Optional[AXValue] = None
    invalid: typing.Optional[bool] = None
    invalid_reason: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.attribute is not None:
            json['attribute'] = self.attribute
        if self.attribute_value is not None:
            json['attributeValue'] = self.attribute_value.to_json()
        if self.superseded is not None:
            json['superseded'] = self.superseded
        if self.native_source is not None:
            json['nativeSource'] = self.native_source.to_json()
        if self.native_source_value is not None:
            json['nativeSourceValue'] = self.native_source_value.to_json()
        if self.invalid is not None:
            json['invalid'] = self.invalid
        if self.invalid_reason is not None:
            json['invalidReason'] = self.invalid_reason
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=AXValueSourceType.from_json(json['type']), value=AXValue.from_json(json['value']) if 'value' in json else None, attribute=str(json['attribute']) if 'attribute' in json else None, attribute_value=AXValue.from_json(json['attributeValue']) if 'attributeValue' in json else None, superseded=bool(json['superseded']) if 'superseded' in json else None, native_source=AXValueNativeSourceType.from_json(json['nativeSource']) if 'nativeSource' in json else None, native_source_value=AXValue.from_json(json['nativeSourceValue']) if 'nativeSourceValue' in json else None, invalid=bool(json['invalid']) if 'invalid' in json else None, invalid_reason=str(json['invalidReason']) if 'invalidReason' in json else None)