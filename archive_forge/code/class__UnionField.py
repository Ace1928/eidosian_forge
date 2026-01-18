import typing
import warnings
import sys
from copy import deepcopy
from dataclasses import MISSING, is_dataclass, fields as dc_fields
from datetime import datetime
from decimal import Decimal
from uuid import UUID
from enum import Enum
from typing_inspect import is_union_type  # type: ignore
from marshmallow import fields, Schema, post_load  # type: ignore
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.core import (_is_supported_generic, _decode_dataclass,
from dataclasses_json.utils import (_is_collection, _is_optional,
class _UnionField(fields.Field):

    def __init__(self, desc, cls, field, *args, **kwargs):
        self.desc = desc
        self.cls = cls
        self.field = field
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if self.allow_none and value is None:
            return None
        for type_, schema_ in self.desc.items():
            if _issubclass_safe(type(value), type_):
                if is_dataclass(value):
                    res = schema_._serialize(value, attr, obj, **kwargs)
                    res['__type'] = str(type_.__name__)
                    return res
                break
            elif isinstance(value, _get_type_origin(type_)):
                return schema_._serialize(value, attr, obj, **kwargs)
        else:
            warnings.warn(f'The type "{type(value).__name__}" (value: "{value}") is not in the list of possible types of typing.Union (dataclass: {self.cls.__name__}, field: {self.field.name}). Value cannot be serialized properly.')
        return super()._serialize(value, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        tmp_value = deepcopy(value)
        if isinstance(tmp_value, dict) and '__type' in tmp_value:
            dc_name = tmp_value['__type']
            for type_, schema_ in self.desc.items():
                if is_dataclass(type_) and type_.__name__ == dc_name:
                    del tmp_value['__type']
                    return schema_._deserialize(tmp_value, attr, data, **kwargs)
        elif isinstance(tmp_value, dict):
            warnings.warn(f'Attempting to deserialize "dict" (value: "{tmp_value}) that does not have a "__type" type specifier field into(dataclass: {self.cls.__name__}, field: {self.field.name}).Deserialization may fail, or deserialization to wrong type may occur.')
            return super()._deserialize(tmp_value, attr, data, **kwargs)
        else:
            for type_, schema_ in self.desc.items():
                if isinstance(tmp_value, _get_type_origin(type_)):
                    return schema_._deserialize(tmp_value, attr, data, **kwargs)
            else:
                warnings.warn(f'The type "{type(tmp_value).__name__}" (value: "{tmp_value}") is not in the list of possible types of typing.Union (dataclass: {self.cls.__name__}, field: {self.field.name}). Value cannot be deserialized properly.')
            return super()._deserialize(tmp_value, attr, data, **kwargs)