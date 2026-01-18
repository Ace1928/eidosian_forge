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
def _parse_type_function(expr: str) -> Tuple[str, List[str]]:
    p = expr.split('(', 1)
    name = p[0].strip()
    assert name.isidentifier(), f'Invalid expression {expr}'
    if len(p) == 1:
        return (name, [])
    arg_expr = p[1].strip().rstrip(')')
    args = [x.strip() for x in arg_expr.split(',')]
    return (name, args)