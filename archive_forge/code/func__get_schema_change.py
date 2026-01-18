import json
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import pandas as pd
import pyarrow as pa
from triad import SerializableRLock
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import cast_pa_table
from .._utils.display import PrettyTable
from ..collections.yielded import Yielded
from ..dataset import (
from ..exceptions import FugueDataFrameOperationError
def _get_schema_change(orig_schema: Optional[Schema], schema: Any) -> Tuple[Schema, List[int]]:
    if orig_schema is None:
        schema = _input_schema(schema).assert_not_empty()
        return (schema, [])
    elif schema is None:
        return (orig_schema.assert_not_empty(), [])
    if isinstance(schema, (str, Schema)) and orig_schema == schema:
        return (orig_schema.assert_not_empty(), [])
    if schema in orig_schema:
        schema = orig_schema.extract(schema).assert_not_empty()
        pos = [orig_schema.index_of_key(x) for x in schema.names]
        if pos == list(range(len(orig_schema))):
            pos = []
        return (schema, pos)
    schema = _input_schema(schema).assert_not_empty()
    pos = [orig_schema.index_of_key(x) for x in schema.names]
    if pos == list(range(len(orig_schema))):
        pos = []
    return (schema, pos)