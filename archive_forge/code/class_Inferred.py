from __future__ import annotations
import collections
import copy
import datetime as dt
import decimal
import ipaddress
import math
import numbers
import typing
import uuid
import warnings
from collections.abc import Mapping as _Mapping
from enum import Enum as EnumType
from marshmallow import class_registry, types, utils, validate
from marshmallow.base import FieldABC, SchemaABC
from marshmallow.exceptions import (
from marshmallow.utils import (
from marshmallow.utils import (
from marshmallow.validate import And, Length
from marshmallow.warnings import RemovedInMarshmallow4Warning
class Inferred(Field):
    """A field that infers how to serialize, based on the value type.

    .. warning::

        This class is treated as private API.
        Users should not need to use this class directly.
    """

    def __init__(self):
        super().__init__()
        self._field_cache = {}

    def _serialize(self, value, attr, obj, **kwargs):
        field_cls = self.root.TYPE_MAPPING.get(type(value))
        if field_cls is None:
            field = super()
        else:
            field = self._field_cache.get(field_cls)
            if field is None:
                field = field_cls()
                field._bind_to_schema(self.name, self.parent)
                self._field_cache[field_cls] = field
        return field._serialize(value, attr, obj, **kwargs)