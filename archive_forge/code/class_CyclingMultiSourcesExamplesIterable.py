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
class CyclingMultiSourcesExamplesIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterables: List[_BaseExamplesIterable], stopping_strategy: Literal['first_exhausted', 'all_exhausted']='first_exhausted'):
        super().__init__()
        self.ex_iterables = ex_iterables
        self.stopping_strategy = stopping_strategy
        self.bool_strategy_func = np.all if stopping_strategy == 'all_exhausted' else np.any

    def _get_indices_iterator(self):
        return cycle(range(len(self.ex_iterables)))

    def __iter__(self):
        iterators = [_HasNextIterator(ex_iterable) for ex_iterable in self.ex_iterables]
        indices_iterator = self._get_indices_iterator()
        is_exhausted = np.full(len(self.ex_iterables), False)
        for i in indices_iterator:
            try:
                yield next(iterators[i])
                if not iterators[i].hasnext():
                    is_exhausted[i] = True
                    if self.bool_strategy_func(is_exhausted):
                        break
                    iterators[i] = _HasNextIterator(self.ex_iterables[i])
            except StopIteration:
                is_exhausted[i] = True
                if self.bool_strategy_func(is_exhausted):
                    break

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'CyclingMultiSourcesExamplesIterable':
        """Shuffle each underlying examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return CyclingMultiSourcesExamplesIterable(ex_iterables, self.stopping_strategy)

    @property
    def n_shards(self) -> int:
        return min((ex_iterable.n_shards for ex_iterable in self.ex_iterables))

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'CyclingMultiSourcesExamplesIterable':
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return CyclingMultiSourcesExamplesIterable([iterable.shard_data_sources(worker_id, num_workers) for iterable in self.ex_iterables], stopping_strategy=self.stopping_strategy)