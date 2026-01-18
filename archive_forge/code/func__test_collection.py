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
def _test_collection(self, value):
    many = self.schema.many or self.many
    if many and (not utils.is_collection(value)):
        raise self.make_error('type', input=value, type=value.__class__.__name__)