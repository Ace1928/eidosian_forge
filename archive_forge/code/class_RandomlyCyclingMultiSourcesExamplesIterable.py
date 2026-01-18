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
class RandomlyCyclingMultiSourcesExamplesIterable(CyclingMultiSourcesExamplesIterable):

    def __init__(self, ex_iterables: List[_BaseExamplesIterable], generator: np.random.Generator, probabilities: Optional[List[float]]=None, stopping_strategy: Literal['first_exhausted', 'all_exhausted']='first_exhausted'):
        super().__init__(ex_iterables, stopping_strategy)
        self.generator = deepcopy(generator)
        self.probabilities = probabilities

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, num_sources: int, random_batch_size=1000, p: Optional[List[float]]=None) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        if p is None:
            while True:
                yield from (int(i) for i in rng.integers(0, num_sources, size=random_batch_size))
        else:
            while True:
                yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=p))

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        return self._iter_random_indices(rng, len(self.ex_iterables), p=self.probabilities)

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'RandomlyCyclingMultiSourcesExamplesIterable':
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return RandomlyCyclingMultiSourcesExamplesIterable(ex_iterables, generator=generator, probabilities=self.probabilities, stopping_strategy=self.stopping_strategy)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'RandomlyCyclingMultiSourcesExamplesIterable':
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return RandomlyCyclingMultiSourcesExamplesIterable([iterable.shard_data_sources(worker_id, num_workers) for iterable in self.ex_iterables], self.generator, self.probabilities, self.stopping_strategy)