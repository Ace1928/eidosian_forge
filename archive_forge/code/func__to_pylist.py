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
def _to_pylist(converter: Callable[[Any], Any], obj: Any, copy: bool=True, str_as_json: bool=True) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str) and str_as_json:
        obj = json.loads(obj)
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if not isinstance(obj, List):
        raise TypeError(f'{obj} is not list')
    if copy:
        return [converter(x) for x in obj]
    else:
        for i in range(len(obj)):
            obj[i] = converter(obj[i])
        return obj