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
class ArrowExamplesIterable(_BaseExamplesIterable):

    def __init__(self, generate_tables_fn: Callable[..., Tuple[Key, pa.Table]], kwargs: dict):
        super().__init__()
        self.generate_tables_fn = generate_tables_fn
        self.kwargs = kwargs
        self.iter_arrow = self._iter_arrow

    def __iter__(self):
        formatter = PythonFormatter()
        for key, pa_table in self.generate_tables_fn(**self.kwargs):
            for pa_subtable in pa_table.to_reader(max_chunksize=config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER):
                formatted_batch = formatter.format_batch(pa_subtable)
                for example in _batch_to_examples(formatted_batch):
                    yield (key, example)

    def _iter_arrow(self):
        yield from self.generate_tables_fn(**self.kwargs)

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'ArrowExamplesIterable':
        return ShuffledDataSourcesArrowExamplesIterable(self.generate_tables_fn, self.kwargs, generator)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'ArrowExamplesIterable':
        """Keep only the requested shard."""
        gen_kwargs_list = _split_gen_kwargs(self.kwargs, max_num_jobs=self.n_shards)
        shard_indices = self.split_shard_indices_by_worker(worker_id, num_workers)
        requested_gen_kwargs = _merge_gen_kwargs([gen_kwargs_list[i] for i in shard_indices])
        return ArrowExamplesIterable(self.generate_tables_fn, requested_gen_kwargs)

    @property
    def n_shards(self) -> int:
        return _number_of_shards_in_gen_kwargs(self.kwargs)