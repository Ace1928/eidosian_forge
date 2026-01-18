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
class ExamplesIterable(_BaseExamplesIterable):

    def __init__(self, generate_examples_fn: Callable[..., Tuple[Key, dict]], kwargs: dict):
        super().__init__()
        self.generate_examples_fn = generate_examples_fn
        self.kwargs = kwargs

    def __iter__(self):
        yield from self.generate_examples_fn(**self.kwargs)

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'ExamplesIterable':
        return ShuffledDataSourcesExamplesIterable(self.generate_examples_fn, self.kwargs, generator)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'ExamplesIterable':
        """Keep only the requested shard."""
        gen_kwargs_list = _split_gen_kwargs(self.kwargs, max_num_jobs=self.n_shards)
        shard_indices = self.split_shard_indices_by_worker(worker_id, num_workers)
        requested_gen_kwargs = _merge_gen_kwargs([gen_kwargs_list[i] for i in shard_indices])
        return ExamplesIterable(self.generate_examples_fn, requested_gen_kwargs)

    @property
    def n_shards(self) -> int:
        return _number_of_shards_in_gen_kwargs(self.kwargs)