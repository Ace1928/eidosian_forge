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
def _parse_types(v: Any):
    if isinstance(v, str):
        return _parse_type(v)
    elif isinstance(v, Dict):
        return pa.struct(_construct_struct(v))
    elif isinstance(v, List):
        if len(v) == 1:
            return pa.list_(_parse_types(v[0]))
        elif len(v) == 3 and v[0] is None:
            return pa.map_(_parse_types(v[1]), _parse_types(v[2]))
        raise SyntaxError(f'{v} is neither a list type nor a map type')
    else:
        raise SyntaxError(f'{v} is not a valid type')