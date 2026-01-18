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
class BufferShuffledExamplesIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterable: _BaseExamplesIterable, buffer_size: int, generator: np.random.Generator):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.buffer_size = buffer_size
        self.generator = generator

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def __iter__(self):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        mem_buffer = []
        for x in self.ex_iterable:
            if len(mem_buffer) == buffer_size:
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x
            else:
                mem_buffer.append(x)
        rng.shuffle(mem_buffer)
        yield from mem_buffer

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'BufferShuffledExamplesIterable':
        """Shuffle the wrapped examples iterable as well as the shuffling buffer."""
        return BufferShuffledExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), buffer_size=self.buffer_size, generator=generator)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'BufferShuffledExamplesIterable':
        """Keep only the requested shard."""
        return BufferShuffledExamplesIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), buffer_size=self.buffer_size, generator=self.generator)

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards