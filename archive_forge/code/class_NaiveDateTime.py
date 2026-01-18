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
class NaiveDateTime(DateTime):
    """A formatted naive datetime string.

    :param format: See :class:`DateTime`.
    :param timezone: Used on deserialization. If `None`,
        aware datetimes are rejected. If not `None`, aware datetimes are
        converted to this timezone before their timezone information is
        removed.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc9
    """
    AWARENESS = 'naive'

    def __init__(self, format: str | None=None, *, timezone: dt.timezone | None=None, **kwargs) -> None:
        super().__init__(format=format, **kwargs)
        self.timezone = timezone

    def _deserialize(self, value, attr, data, **kwargs) -> dt.datetime:
        ret = super()._deserialize(value, attr, data, **kwargs)
        if is_aware(ret):
            if self.timezone is None:
                raise self.make_error('invalid_awareness', awareness=self.AWARENESS, obj_type=self.OBJ_TYPE)
            ret = ret.astimezone(self.timezone).replace(tzinfo=None)
        return ret