import copy
import json
import sys
import warnings
from collections import defaultdict, namedtuple
from dataclasses import (MISSING,
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (Any, Collection, Mapping, Union, get_type_hints,
from uuid import UUID
from typing_inspect import is_union_type  # type: ignore
from dataclasses_json import cfg
from dataclasses_json.utils import (_get_type_cons, _get_type_origin,
def _encode_json_type(value, default=_ExtendedEncoder().default):
    if isinstance(value, Json.__args__):
        if isinstance(value, list):
            return [_encode_json_type(i) for i in value]
        elif isinstance(value, dict):
            return {k: _encode_json_type(v) for k, v in value.items()}
        else:
            return value
    return default(value)