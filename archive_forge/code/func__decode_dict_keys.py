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
def _decode_dict_keys(key_type, xs, infer_missing):
    """
    Because JSON object keys must be strs, we need the extra step of decoding
    them back into the user's chosen python type
    """
    decode_function = key_type
    if key_type is None or key_type == Any or isinstance(key_type, TypeVar):
        decode_function = key_type = lambda x: x
    elif _get_type_origin(key_type) in {tuple, Tuple}:
        decode_function = tuple
        key_type = key_type
    return map(decode_function, _decode_items(key_type, xs, infer_missing))