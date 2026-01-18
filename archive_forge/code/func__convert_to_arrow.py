import copy
import itertools
import sys
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import cycle, islice
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from . import config
from .arrow_dataset import Dataset, DatasetInfoMixin
from .features import Features
from .features.features import FeatureType, _align_features, _check_if_features_can_be_aligned, cast_to_python_objects
from .filesystems import _reset_fsspec_lock
from .formatting import PythonFormatter, TensorFormatter, get_format_type_from_alias, get_formatter
from .info import DatasetInfo
from .splits import NamedSplit
from .table import cast_table_to_features, read_schema_from_file, table_cast
from .utils.logging import get_logger
from .utils.py_utils import Literal
from .utils.sharding import _merge_gen_kwargs, _number_of_shards_in_gen_kwargs, _shuffle_gen_kwargs, _split_gen_kwargs
def _convert_to_arrow(iterable: Iterable[Tuple[Key, dict]], batch_size: int, drop_last_batch: bool=False) -> Iterator[Tuple[Key, pa.Table]]:
    """Convert and group examples in Arrow tables of size `batch_size`.

    Args:
        iterable (`Iterable[Tuple[Key, dict]]`):
            An examples iterable containing tuples (example_key, example) of type (int/str, dict)
        batch_size (`Optional[int]`):
            Size of each sub-table to yield. If None or <= 0, yields the full table.
        drop_last_batch (`bool`, defaults to `False`):
            Drop the last batch if it is smaller than `batch_size`.
    """
    if batch_size is None or batch_size <= 0:
        yield ('all', pa.Table.from_pylist(cast_to_python_objects([example for _, example in iterable], only_1d_for_numpy=True)))
        return
    iterator = iter(iterable)
    for key, example in iterator:
        iterator_batch = islice(iterator, batch_size - 1)
        key_examples_list = [(key, example)] + list(iterator_batch)
        if len(key_examples_list) < batch_size and drop_last_batch:
            return
        keys, examples = zip(*key_examples_list)
        new_key = '_'.join((str(key) for key in keys))
        yield (new_key, pa.Table.from_pylist(cast_to_python_objects(examples, only_1d_for_numpy=True)))