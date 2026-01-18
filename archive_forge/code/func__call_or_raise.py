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
def _call_or_raise(self, func, value, attr):
    if len(utils.get_func_args(func)) > 1:
        if self.parent.context is None:
            msg = f'No context available for Function field {attr!r}'
            raise ValidationError(msg)
        return func(value, self.parent.context)
    return func(value)