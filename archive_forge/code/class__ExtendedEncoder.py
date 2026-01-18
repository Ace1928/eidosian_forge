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
class _ExtendedEncoder(json.JSONEncoder):

    def default(self, o) -> Json:
        result: Json
        if _isinstance_safe(o, Collection):
            if _isinstance_safe(o, Mapping):
                result = dict(o)
            else:
                result = list(o)
        elif _isinstance_safe(o, datetime):
            result = o.timestamp()
        elif _isinstance_safe(o, UUID):
            result = str(o)
        elif _isinstance_safe(o, Enum):
            result = o.value
        elif _isinstance_safe(o, Decimal):
            result = str(o)
        else:
            result = json.JSONEncoder.default(self, o)
        return result