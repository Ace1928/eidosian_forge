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
def _replace_field(field: pa.Field, is_type: Callable[[pa.DataType], bool], convert_type: Callable[[pa.DataType], pa.DataType], recursive: bool):
    old_type = field.type
    new_type = replace_type(old_type, is_type, convert_type, recursive=recursive)
    if old_type is new_type or old_type == new_type:
        return field
    return pa.field(field.name, new_type, nullable=field.nullable, metadata=field.metadata)