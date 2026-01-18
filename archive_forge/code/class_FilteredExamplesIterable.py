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
class FilteredExamplesIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterable: _BaseExamplesIterable, function: Callable, with_indices: bool=False, input_columns: Optional[List[str]]=None, batched: bool=False, batch_size: Optional[int]=1000, fn_kwargs: Optional[dict]=None, formatting: Optional['FormattingConfig']=None, format_type='deprecated'):
        if format_type != 'deprecated':
            warning_msg = "'format_type' is deprecated and will be removed in the next major version of datasets. "
            help_message = "Please use 'formatting=FormattingConfig(format_type=format_type)' instead."
            warnings.warn(warning_msg + help_message, category=FutureWarning, stacklevel=2)
            formatting = FormattingConfig(format_type=format_type)
        super().__init__()
        self.ex_iterable = ex_iterable
        self.function = function
        self.batched = batched
        self.batch_size = batch_size
        self.with_indices = with_indices
        self.input_columns = input_columns
        self.fn_kwargs = fn_kwargs or {}
        self.formatting = formatting
        if self.formatting and self.formatting.format_type == 'arrow':
            self.iter_arrow = self._iter_arrow

    def __iter__(self):
        if self.formatting and self.formatting.format_type == 'arrow':
            yield from ArrowExamplesIterable(self._iter_arrow, {})
        else:
            yield from self._iter()

    def _iter(self):
        if self.formatting:
            formatter = get_formatter(self.formatting.format_type)
            format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
        else:
            format_dict = None
        iterator = iter(self.ex_iterable)
        current_idx = 0
        if self.batched:
            for key, example in iterator:
                iterator_batch = iterator if self.batch_size is None or self.batch_size <= 0 else islice(iterator, self.batch_size - 1)
                key_examples_list = [(key, example)] + list(iterator_batch)
                keys, examples = zip(*key_examples_list)
                batch = _examples_to_batch(examples)
                batch = format_dict(batch) if format_dict else batch
                inputs = batch
                function_args = [inputs] if self.input_columns is None else [inputs[col] for col in self.input_columns]
                if self.with_indices:
                    function_args.append([current_idx + i for i in range(len(key_examples_list))])
                mask = self.function(*function_args, **self.fn_kwargs)
                for key_example, to_keep in zip(key_examples_list, mask):
                    if to_keep:
                        yield key_example
                    current_idx += 1
        else:
            for key, example in iterator:
                example = dict(example)
                inputs = format_dict(example) if format_dict else example
                function_args = [inputs] if self.input_columns is None else [inputs[col] for col in self.input_columns]
                if self.with_indices:
                    function_args.append(current_idx)
                to_keep = self.function(*function_args, **self.fn_kwargs)
                if to_keep:
                    yield (key, example)
                current_idx += 1

    def _iter_arrow(self):
        if self.ex_iterable.iter_arrow:
            iterator = _batch_arrow_tables(self.ex_iterable.iter_arrow(), batch_size=self.batch_size if self.batched else 1)
        else:
            iterator = _convert_to_arrow(self.ex_iterable, batch_size=self.batch_size if self.batched else 1)
        current_idx = 0
        for key, pa_table in iterator:
            function_args = [pa_table] if self.input_columns is None else [pa_table[col] for col in self.input_columns]
            if self.with_indices:
                if self.batched:
                    function_args.append([current_idx + i for i in range(len(pa_table))])
                else:
                    function_args.append(current_idx)
            mask = self.function(*function_args, **self.fn_kwargs)
            if self.batched:
                yield (key, pa_table.filter(mask))
            elif mask.as_py() if isinstance(mask, pa.BooleanScalar) else mask:
                yield (key, pa_table)
            current_idx += len(pa_table)

    def shuffle_data_sources(self, seed: Optional[int]) -> 'FilteredExamplesIterable':
        """Shuffle the wrapped examples iterable."""
        return FilteredExamplesIterable(self.ex_iterable.shuffle_data_sources(seed), function=self.function, with_indices=self.with_indices, input_columns=self.input_columns, batched=self.batched, batch_size=self.batch_size)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'FilteredExamplesIterable':
        """Keep only the requested shard."""
        return FilteredExamplesIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), function=self.function, with_indices=self.with_indices, input_columns=self.input_columns, batched=self.batched, batch_size=self.batch_size)

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards