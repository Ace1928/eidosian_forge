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
def _date_eq(o1: Any, o2: Any) -> bool:
    if o1 == o2:
        return True
    nat1 = o1 is pd.NaT or o1 is None
    nat2 = o2 is pd.NaT or o2 is None
    if nat1 and nat2:
        return True
    if nat1 or nat2:
        return False
    return o1.year == o2.year and o1.month == o2.month and (o1.day == o2.day)