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
class SchemaedDataPartitioner:
    """Partitioner for stream of array like data with given schema.
    It uses :func"`~triad.utils.iter.Slicer` to partition the stream

    :param schema: the schema of the data stream to process
    :param key_positions: positions of partition keys on `schema`
    :param sizer: the function to get size of an item
    :param row_limit: max row for each slice, defaults to None
    :param size_limit: max byte size for each slice, defaults to None
    """

    def __init__(self, schema: pa.Schema, key_positions: List[int], sizer: Optional[Callable[[Any], int]]=None, row_limit: int=0, size_limit: Any=None):
        self._eq_funcs: List[Any] = [None] * len(schema)
        self._keys = key_positions
        for p in key_positions:
            self._eq_funcs[p] = get_eq_func(schema.types[p])
        self._slicer = Slicer(sizer=sizer, row_limit=row_limit, size_limit=size_limit, slicer=self._is_boundary)
        self._hitting_boundary = True

    def partition(self, data: Iterable[Any]) -> Iterable[Tuple[int, int, EmptyAwareIterable[Any]]]:
        """Partition the given data stream

        :param data: iterable of array like objects
        :yield: iterable of <partition_no, slice_no, slice iterable> tuple
        """
        self._hitting_boundary = False
        slice_no = 0
        partition_no = 0
        for slice_ in self._slicer.slice(data):
            if self._hitting_boundary:
                slice_no = 0
                partition_no += 1
                self._hitting_boundary = False
            yield (partition_no, slice_no, slice_)
            slice_no += 1

    def _is_boundary(self, no: int, current: Any, last: Any) -> bool:
        self._hitting_boundary = any((not self._eq_funcs[p](current[p], last[p]) for p in self._keys))
        return self._hitting_boundary