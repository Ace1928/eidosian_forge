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
def _format_num(self, value):
    num = decimal.Decimal(str(value))
    if self.allow_nan:
        if num.is_nan():
            return decimal.Decimal('NaN')
    if self.places is not None and num.is_finite():
        num = num.quantize(self.places, rounding=self.rounding)
    return num