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
class _TimestampField(fields.Field):

    def _serialize(self, value, attr, obj, **kwargs):
        if value is not None:
            return value.timestamp()
        elif not self.required:
            return None
        else:
            raise ValidationError(self.default_error_messages['required'])

    def _deserialize(self, value, attr, data, **kwargs):
        if value is not None:
            return _timestamp_to_dt_aware(value)
        elif not self.required:
            return None
        else:
            raise ValidationError(self.default_error_messages['required'])