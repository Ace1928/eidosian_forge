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
class StepExamplesIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterable: _BaseExamplesIterable, step: int, offset: int):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.step = step
        self.offset = offset

    def __iter__(self):
        ex_iterator = iter(self.ex_iterable)
        while True:
            batch = list(islice(ex_iterator, self.step))
            if len(batch) > self.offset:
                yield batch[self.offset]
            else:
                break

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'StepExamplesIterable':
        return StepExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), step=self.step, offset=self.offset)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'StepExamplesIterable':
        return StepExamplesIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), step=self.step, offset=self.offset)

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards