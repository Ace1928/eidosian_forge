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
def _concatenate_iterable_datasets(dsets: List[IterableDataset], info: Optional[DatasetInfo]=None, split: Optional[NamedSplit]=None, axis: int=0) -> IterableDataset:
    """
    Converts a list of `IterableDataset` with the same schema into a single `IterableDataset`.
    Missing data are filled with None values.

    <Added version="2.4.0"/>

    Args:
        dsets (`List[datasets.IterableDataset]`): List of Datasets to concatenate.
        info (`DatasetInfo`, optional): Dataset information, like description, citation, etc.
        split (`NamedSplit`, optional): Name of the dataset split.
        axis (``{0, 1}``, default ``0``, meaning over rows):
            Axis to concatenate over, where ``0`` means over rows (vertically) and ``1`` means over columns
            (horizontally).

            *New in version 1.6.0*

    Example:

    ```py
    >>> ds3 = _concatenate_iterable_datasets([ds1, ds2])
    ```
    """
    dsets = [d._resolve_features() for d in dsets]
    if axis == 0:
        _check_if_features_can_be_aligned([dset.features for dset in dsets])
    else:
        _check_column_names([col_name for dset in dsets for col_name in dset.features])
    features = Features({k: v for features in _align_features([dset.features for dset in dsets]) for k, v in features.items()})
    ex_iterables = [d._ex_iterable for d in dsets]
    if axis == 0:
        ex_iterable = VerticallyConcatenatedMultiSourcesExamplesIterable(ex_iterables)
    else:
        ex_iterable = HorizontallyConcatenatedMultiSourcesExamplesIterable(ex_iterables)
    if info is None:
        info = DatasetInfo.from_merge([d.info for d in dsets])
    else:
        info = info.copy()
    info.features = features
    token_per_repo_id = {repo_id: token for dataset in dsets for repo_id, token in dataset._token_per_repo_id.items()}
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=split, token_per_repo_id=token_per_repo_id)