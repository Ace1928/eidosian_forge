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
class _TupleVarLen(fields.List):
    """
    variable-length homogeneous tuples
    """

    def _deserialize(self, value, attr, data, **kwargs):
        optional_list = super()._deserialize(value, attr, data, **kwargs)
        return None if optional_list is None else tuple(optional_list)