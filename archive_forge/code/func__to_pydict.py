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
def _to_pydict(schema: Dict[str, Callable[[Any], Any]], obj: Any, str_as_json: bool=True) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str) and str_as_json:
        obj = json.loads(obj)
    if not isinstance(obj, Dict):
        raise TypeError(f'{obj} is not dict')
    result: Dict[str, Any] = {}
    for k, v in schema.items():
        result[k] = v(obj.get(k, None))
    return result