import json
import pickle
from datetime import date, datetime
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow.compute import CastOptions, binary_join_element_wise
from pyarrow.json import read_json, ParseOptions as JsonParseOptions
from triad.constants import TRIAD_VAR_QUOTE
from .convert import as_type
from .iter import EmptyAwareIterable, Slicer
from .json import loads_no_dup
from .schema import move_to_unquoted, quote_name, unquote_name
from .assertion import assert_or_throw
def _parse_tokens(expr: str) -> Iterable[str]:
    expr += ','
    last = 0
    skip = False
    i = 0
    while i < len(expr):
        if expr[i] == TRIAD_VAR_QUOTE:
            e = move_to_unquoted(expr, i)
            s = unquote_name(expr[i:e])
            yield ('"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"')
            last = i = e
            continue
        if expr[i] == '(':
            skip = True
        if expr[i] == ')':
            skip = False
        if not skip and expr[i] in _SPECIAL_TOKENS:
            s = expr[last:i].strip()
            if s != '':
                yield ('"' + s + '"')
            if expr[i] == '<':
                yield '[null,'
            elif expr[i] == '>':
                yield ']'
            else:
                yield expr[i]
            last = i + 1
        i += 1