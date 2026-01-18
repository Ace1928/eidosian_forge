import contextlib
import copy
import fnmatch
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial, wraps
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from random import sample
from typing import (
from typing import Sequence as Sequence_
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from huggingface_hub import CommitInfo, CommitOperationAdd, CommitOperationDelete, DatasetCard, DatasetCardData, HfApi
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter, OptimizedTypedSequence
from .data_files import sanitize_patterns
from .download.streaming_download_manager import xgetsize
from .features import Audio, ClassLabel, Features, Image, Sequence, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .fingerprint import (
from .formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from .formatting.formatting import LazyDict, _is_range_contiguous
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .search import IndexableMixin
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import (
from .tasks import TaskTemplate
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import estimate_dataset_size
from .utils.hub import list_files_info, preupload_lfs_files
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike
@staticmethod
def _map_single(shard: 'Dataset', function: Optional[Callable]=None, with_indices: bool=False, with_rank: bool=False, input_columns: Optional[List[str]]=None, batched: bool=False, batch_size: Optional[int]=1000, drop_last_batch: bool=False, remove_columns: Optional[List[str]]=None, keep_in_memory: bool=False, cache_file_name: Optional[str]=None, writer_batch_size: Optional[int]=1000, features: Optional[Features]=None, disable_nullable: bool=False, fn_kwargs: Optional[dict]=None, new_fingerprint: Optional[str]=None, rank: Optional[int]=None, offset: int=0) -> Iterable[Tuple[int, bool, Union[int, 'Dataset']]]:
    """Apply a function to all the elements in the table (individually or in batches)
        and update the table (if function does update examples).

        Args:
            shard (`datasets.Dataset`): Dataset to map the transform on.
            function (`Callable`): with one of the following signature:
                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> Dict[str, Any]` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> Dict[str, List]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: lambda x: x
            with_indices (`bool`, defaults to `False`): Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, default `False`): Provide process rank to `function`. Note that in this case the signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`Optional[List[str]]`, defaults to `None`): The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`): Provide batch of examples to `function`
            batch_size (`int`, optional, defaults to `1000`): Number of examples per batch provided to `function` if `batched=True`
                `batch_size <= 0` or `batch_size == None`: Provide the full dataset as a single batch to `function`
            drop_last_batch (`bool`, default: `False`): Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`Optional[List[str]]`, defaults to `None`): Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`): Keep the dataset in memory instead of writing it to a cache file.
            cache_file_name (`str`, optional, defaults to `None`): Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, default `1000`): Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
            features (`Optional[datasets.Features]`, defaults to `None`): Use a specific Features to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`): Disallow null values in the table.
            fn_kwargs (`Dict`, optional, defaults to `None`): Keyword arguments to be passed to `function`
            new_fingerprint (`str`, optional, defaults to `None`): the new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
            rank: (`int`, optional, defaults to `None`): If specified, this is the process rank when doing multiprocessing
            offset: (`int`, defaults to 0): If specified, this is an offset applied to the indices passed to `function` if `with_indices=True`.
        """
    if fn_kwargs is None:
        fn_kwargs = {}
    if batched and (batch_size is None or batch_size <= 0):
        batch_size = shard.num_rows
    update_data = None
    format_kwargs = shard._format_kwargs.copy()
    if not input_columns and shard._format_type is None:
        format_kwargs['lazy'] = True
    input_formatter = get_formatter(shard._format_type, features=shard.features, **format_kwargs)

    class NumExamplesMismatchError(Exception):
        pass

    def validate_function_output(processed_inputs, indices):
        """Validate output of the map function."""
        if processed_inputs is not None and (not isinstance(processed_inputs, (Mapping, pa.Table, pd.DataFrame))):
            raise TypeError(f'Provided `function` which is applied to all elements of table returns a variable of type {type(processed_inputs)}. Make sure provided `function` returns a variable of type `dict` (or a pyarrow table) to update the dataset or `None` if you are only interested in side effects.')
        elif isinstance(indices, list) and isinstance(processed_inputs, Mapping):
            allowed_batch_return_types = (list, np.ndarray, pd.Series)
            if config.TF_AVAILABLE and 'tensorflow' in sys.modules:
                import tensorflow as tf
                allowed_batch_return_types += (tf.Tensor,)
            if config.TORCH_AVAILABLE and 'torch' in sys.modules:
                import torch
                allowed_batch_return_types += (torch.Tensor,)
            if config.JAX_AVAILABLE and 'jax' in sys.modules:
                import jax.numpy as jnp
                allowed_batch_return_types += (jnp.ndarray,)
            all_dict_values_are_lists = all((isinstance(value, allowed_batch_return_types) for value in processed_inputs.values()))
            if all_dict_values_are_lists is False:
                raise TypeError(f'Provided `function` which is applied to all elements of table returns a `dict` of types {[type(x) for x in processed_inputs.values()]}. When using `batched=True`, make sure provided `function` returns a `dict` of types like `{allowed_batch_return_types}`.')

    def apply_function_on_filtered_inputs(pa_inputs, indices, check_same_num_examples=False, offset=0):
        """Utility to apply the function on a selection of columns."""
        nonlocal update_data
        inputs = format_table(pa_inputs, 0 if not batched else range(pa_inputs.num_rows), format_columns=input_columns, formatter=input_formatter)
        fn_args = [inputs] if input_columns is None else [inputs[col] for col in input_columns]
        if offset == 0:
            effective_indices = indices
        else:
            effective_indices = [i + offset for i in indices] if isinstance(indices, list) else indices + offset
        additional_args = ()
        if with_indices:
            additional_args += (effective_indices,)
        if with_rank:
            additional_args += (rank,)
        processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
        if isinstance(processed_inputs, LazyDict):
            processed_inputs = {k: v for k, v in processed_inputs.data.items() if k not in processed_inputs.keys_to_format}
            returned_lazy_dict = True
        else:
            returned_lazy_dict = False
        if update_data is None:
            update_data = isinstance(processed_inputs, (Mapping, pa.Table, pd.DataFrame))
            validate_function_output(processed_inputs, indices)
        if not update_data:
            return None
        if shard._format_type or input_columns:
            inputs_to_merge = dict(zip(pa_inputs.column_names, pa_inputs.itercolumns()))
        elif isinstance(inputs, LazyDict):
            inputs_to_merge = {k: v if k not in inputs.keys_to_format else pa_inputs[k] for k, v in inputs.data.items()}
        else:
            inputs_to_merge = inputs
        if remove_columns is not None:
            for column in remove_columns:
                if column in inputs_to_merge:
                    inputs_to_merge.pop(column)
                if returned_lazy_dict and column in processed_inputs:
                    processed_inputs.pop(column)
        if check_same_num_examples:
            input_num_examples = len(pa_inputs)
            processed_inputs_num_examples = len(processed_inputs[next(iter(processed_inputs.keys()))])
            if input_num_examples != processed_inputs_num_examples:
                raise NumExamplesMismatchError()
        if isinstance(inputs, Mapping) and isinstance(processed_inputs, Mapping):
            return {**inputs_to_merge, **processed_inputs}
        else:
            return processed_inputs

    def init_buffer_and_writer():
        writer_features = features
        if writer_features is None:
            writer_features = shard.features
            update_features = True
        else:
            update_features = False
        if keep_in_memory or cache_file_name is None:
            buf_writer = pa.BufferOutputStream()
            tmp_file = None
            writer = ArrowWriter(features=writer_features, stream=buf_writer, writer_batch_size=writer_batch_size, update_features=update_features, fingerprint=new_fingerprint, disable_nullable=disable_nullable)
        else:
            buf_writer = None
            logger.info(f'Caching processed dataset at {cache_file_name}')
            tmp_file = tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(cache_file_name), delete=False)
            writer = ArrowWriter(features=writer_features, path=tmp_file.name, writer_batch_size=writer_batch_size, update_features=update_features, fingerprint=new_fingerprint, disable_nullable=disable_nullable)
        return (buf_writer, writer, tmp_file)
    num_examples_progress_update = 0
    buf_writer, writer, tmp_file = (None, None, None)
    with contextlib.ExitStack() as stack:
        try:
            arrow_formatted_shard = shard.with_format('arrow')
            if not batched:
                shard_iterable = enumerate(arrow_formatted_shard)
            else:
                num_rows = len(shard) if not drop_last_batch else len(shard) // batch_size * batch_size
                shard_iterable = zip(range(0, num_rows, batch_size), arrow_formatted_shard.iter(batch_size, drop_last_batch=drop_last_batch))
            if not batched:
                _time = time.time()
                for i, example in shard_iterable:
                    example = apply_function_on_filtered_inputs(example, i, offset=offset)
                    if update_data:
                        if i == 0:
                            buf_writer, writer, tmp_file = init_buffer_and_writer()
                            stack.enter_context(writer)
                        if isinstance(example, pa.Table):
                            writer.write_row(example)
                        elif isinstance(example, pd.DataFrame):
                            writer.write_row(pa.Table.from_pandas(example))
                        else:
                            writer.write(example)
                    num_examples_progress_update += 1
                    if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield (rank, False, num_examples_progress_update)
                        num_examples_progress_update = 0
            else:
                _time = time.time()
                for i, batch in shard_iterable:
                    num_examples_in_batch = len(batch)
                    indices = list(range(*slice(i, i + batch_size).indices(shard.num_rows)))
                    try:
                        batch = apply_function_on_filtered_inputs(batch, indices, check_same_num_examples=len(shard.list_indexes()) > 0, offset=offset)
                    except NumExamplesMismatchError:
                        raise DatasetTransformationNotAllowedError("Using `.map` in batched mode on a dataset with attached indexes is allowed only if it doesn't create or remove existing examples. You can first run `.drop_index() to remove your index and then re-add it.") from None
                    if update_data:
                        if i == 0:
                            buf_writer, writer, tmp_file = init_buffer_and_writer()
                            stack.enter_context(writer)
                        if isinstance(batch, pa.Table):
                            writer.write_table(batch)
                        elif isinstance(batch, pd.DataFrame):
                            writer.write_table(pa.Table.from_pandas(batch))
                        else:
                            writer.write_batch(batch)
                    num_examples_progress_update += num_examples_in_batch
                    if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield (rank, False, num_examples_progress_update)
                        num_examples_progress_update = 0
            if update_data and writer is not None:
                writer.finalize()
        except (Exception, KeyboardInterrupt):
            yield (rank, False, num_examples_progress_update)
            if update_data:
                if writer is not None:
                    writer.finalize()
                if tmp_file is not None:
                    tmp_file.close()
                    if os.path.exists(tmp_file.name):
                        os.remove(tmp_file.name)
            raise
    yield (rank, False, num_examples_progress_update)
    if update_data and tmp_file is not None:
        tmp_file.close()
        shutil.move(tmp_file.name, cache_file_name)
        umask = os.umask(438)
        os.umask(umask)
        os.chmod(cache_file_name, 438 & ~umask)
    if update_data:
        info = shard.info.copy()
        info.features = writer._features
        info.task_templates = None
        if buf_writer is None:
            yield (rank, True, Dataset.from_file(cache_file_name, info=info, split=shard.split))
        else:
            yield (rank, True, Dataset.from_buffer(buf_writer.getvalue(), info=info, split=shard.split))
    else:
        yield (rank, True, shard)