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
class IP(Field):
    """A IP address field.

    :param bool exploded: If `True`, serialize ipv6 address in long form, ie. with groups
        consisting entirely of zeros included.

    .. versionadded:: 3.8.0
    """
    default_error_messages = {'invalid_ip': 'Not a valid IP address.'}
    DESERIALIZATION_CLASS = None

    def __init__(self, *args, exploded=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploded = exploded

    def _serialize(self, value, attr, obj, **kwargs) -> str | None:
        if value is None:
            return None
        if self.exploded:
            return value.exploded
        return value.compressed

    def _deserialize(self, value, attr, data, **kwargs) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        if value is None:
            return None
        try:
            return (self.DESERIALIZATION_CLASS or ipaddress.ip_address)(utils.ensure_text_type(value))
        except (ValueError, TypeError) as error:
            raise self.make_error('invalid_ip') from error