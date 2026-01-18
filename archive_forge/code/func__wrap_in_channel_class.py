from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def _wrap_in_channel_class(obj, encoding):
    if isinstance(obj, SchemaBase):
        return obj
    if isinstance(obj, str):
        obj = {'shorthand': obj}
    if isinstance(obj, (list, tuple)):
        return [_wrap_in_channel_class(subobj, encoding) for subobj in obj]
    if encoding not in name_to_channel:
        warnings.warn("Unrecognized encoding channel '{}'".format(encoding), stacklevel=1)
        return obj
    classes = name_to_channel[encoding]
    cls = classes['value'] if 'value' in obj else classes['field']
    try:
        return cls.from_dict(obj, validate=False)
    except jsonschema.ValidationError:
        return obj