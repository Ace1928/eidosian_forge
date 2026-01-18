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
class TypedExamplesIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterable: _BaseExamplesIterable, features: Features, token_per_repo_id: Dict[str, Union[str, bool, None]]):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.features = features
        self.token_per_repo_id = token_per_repo_id
        if self.ex_iterable.iter_arrow is not None:
            self.iter_arrow = self._iter_arrow

    def __iter__(self):
        for key, example in self.ex_iterable:
            yield (key, _apply_feature_types_on_example(example, self.features, token_per_repo_id=self.token_per_repo_id))

    def _iter_arrow(self) -> Iterator[Tuple[Key, pa.Table]]:
        schema = self.features.arrow_schema
        for key, pa_table in self.ex_iterable.iter_arrow():
            columns = set(pa_table.column_names)
            for column_name in self.features:
                if column_name not in columns:
                    col = pa.NullArray.from_buffers(pa.null(), len(pa_table), [None])
                    pa_table = pa_table.append_column(column_name, col)
            if pa_table.schema != schema:
                pa_table = cast_table_to_features(pa_table, self.features)
            yield (key, pa_table)

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'TypedExamplesIterable':
        """Shuffle the wrapped examples iterable."""
        return TypedExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), features=self.features, token_per_repo_id=self.token_per_repo_id)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'TypedExamplesIterable':
        """Keep only the requested shard."""
        return TypedExamplesIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), features=self.features, token_per_repo_id=self.token_per_repo_id)

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards