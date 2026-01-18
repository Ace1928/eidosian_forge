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
class VerticallyConcatenatedMultiSourcesExamplesIterable(_BaseExamplesIterable):
    """
    VerticallyConcatenatedMultiSourcesExamplesIterable simply chains the input iterables.
    It doesn't require the examples iterables to always yield the same columns.
    Instead, this is handled by the `IterableDataset` class or `TypedExamplesIterable`.

    For information, `IterableDataset` merges the features of all the datasets to concatenate into one.
    We use `IterableDataset._resolve_features` to obtain the features of all the datasets to concatenate.

    Then for each example, `IterableDataset` and `TypedExamplesIterable` automatically fill missing columns with None.
    This is done with `_apply_feature_types_on_example`.
    """

    def __init__(self, ex_iterables: List[_BaseExamplesIterable]):
        super().__init__()
        self.ex_iterables = ex_iterables
        if all((ex_iterable.iter_arrow is not None for ex_iterable in ex_iterables)):
            self.iter_arrow = self._iter_arrow

    def __iter__(self):
        for ex_iterable in self.ex_iterables:
            yield from ex_iterable

    def _iter_arrow(self):
        for ex_iterable in self.ex_iterables:
            yield from ex_iterable.iter_arrow()

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'VerticallyConcatenatedMultiSourcesExamplesIterable':
        """Shuffle the list of examples iterable, as well as each underlying examples iterable."""
        rng = deepcopy(generator)
        ex_iterables = list(self.ex_iterables)
        rng.shuffle(ex_iterables)
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in ex_iterables]
        return VerticallyConcatenatedMultiSourcesExamplesIterable(ex_iterables)

    @property
    def n_shards(self) -> int:
        return min((ex_iterable.n_shards for ex_iterable in self.ex_iterables))

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'VerticallyConcatenatedMultiSourcesExamplesIterable':
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return VerticallyConcatenatedMultiSourcesExamplesIterable([iterable.shard_data_sources(worker_id, num_workers) for iterable in self.ex_iterables])