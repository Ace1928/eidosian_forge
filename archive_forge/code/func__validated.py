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
def _validated(self, value):
    try:
        num = super()._validated(value)
    except decimal.InvalidOperation as error:
        raise self.make_error('invalid') from error
    if not self.allow_nan and (num.is_nan() or num.is_infinite()):
        raise self.make_error('special')
    return num