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
class TensorflowDatasetMixin:
    _TF_DATASET_REFS = set()

    @staticmethod
    def _get_output_signature(dataset: 'Dataset', collate_fn: Callable, collate_fn_args: dict, cols_to_retain: Optional[List[str]]=None, batch_size: Optional[int]=None, num_test_batches: int=20):
        """Private method used by `to_tf_dataset()` to find the shapes and dtypes of samples from this dataset
           after being passed through the collate_fn. Tensorflow needs an exact signature for tf.numpy_function, so
           the only way to do this is to run test batches - the collator may add or rename columns, so we can't figure
           it out just by inspecting the dataset.

        Args:
            dataset (`Dataset`): Dataset to load samples from.
            collate_fn(`bool`): Shuffle the dataset order when loading. Recommended True for training, False for
                validation/evaluation.
            collate_fn(`Callable`): A function or callable object (such as a `DataCollator`) that will collate
                lists of samples into a batch.
            collate_fn_args (`Dict`): A `dict` of keyword arguments to be passed to the
                `collate_fn`.
            batch_size (`int`, optional): The size of batches loaded from the dataset. Used for shape inference.
                Can be None, which indicates that batch sizes can be variable.
            num_test_batches (`int`): The number of batches to load from the dataset for shape inference.

        Returns:
            `dict`: Dict mapping column names to tf.Tensorspec objects
            `dict`: Dict mapping column names to np.dtype objects
        """
        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError('Called a Tensorflow-specific function but Tensorflow is not installed.')
        if len(dataset) == 0:
            raise ValueError('Unable to get the output signature because the dataset is empty.')
        if batch_size is not None:
            batch_size = min(len(dataset), batch_size)
        test_batch_size = 1
        if cols_to_retain is not None:
            cols_to_retain = list(set(cols_to_retain + ['label_ids', 'label', 'labels']))
        test_batches = []
        for _ in range(num_test_batches):
            indices = sample(range(len(dataset)), test_batch_size)
            test_batch = dataset[indices]
            if cols_to_retain is not None:
                test_batch = {key: value for key, value in test_batch.items() if key in cols_to_retain}
            test_batch = [{key: value[i] for key, value in test_batch.items()} for i in range(test_batch_size)]
            test_batch = collate_fn(test_batch, **collate_fn_args)
            test_batches.append(test_batch)
        tf_columns_to_signatures = {}
        np_columns_to_dtypes = {}
        for column in test_batches[0].keys():
            raw_arrays = [batch[column] for batch in test_batches]
            np_arrays = []
            for array in raw_arrays:
                if isinstance(array, np.ndarray):
                    np_arrays.append(array)
                elif isinstance(array, tf.Tensor):
                    np_arrays.append(array.numpy())
                else:
                    np_arrays.append(np.array(array))
            if np.issubdtype(np_arrays[0].dtype, np.integer) or np_arrays[0].dtype == bool:
                tf_dtype = tf.int64
                np_dtype = np.int64
            elif np.issubdtype(np_arrays[0].dtype, np.number):
                tf_dtype = tf.float32
                np_dtype = np.float32
            elif np_arrays[0].dtype.kind == 'U':
                np_dtype = np.unicode_
                tf_dtype = tf.string
            else:
                raise RuntimeError(f'Unrecognized array dtype {np_arrays[0].dtype}. \nNested types and image/audio types are not supported yet.')
            shapes = [array.shape for array in np_arrays]
            static_shape = []
            for dim in range(len(shapes[0])):
                sizes = {shape[dim] for shape in shapes}
                if dim == 0:
                    static_shape.append(batch_size)
                    continue
                if len(sizes) == 1:
                    static_shape.append(sizes.pop())
                else:
                    static_shape.append(None)
            tf_columns_to_signatures[column] = tf.TensorSpec(shape=static_shape, dtype=tf_dtype)
            np_columns_to_dtypes[column] = np_dtype
        return (tf_columns_to_signatures, np_columns_to_dtypes)

    def to_tf_dataset(self, batch_size: Optional[int]=None, columns: Optional[Union[str, List[str]]]=None, shuffle: bool=False, collate_fn: Optional[Callable]=None, drop_remainder: bool=False, collate_fn_args: Optional[Dict[str, Any]]=None, label_cols: Optional[Union[str, List[str]]]=None, prefetch: bool=True, num_workers: int=0, num_test_batches: int=20):
        """Create a `tf.data.Dataset` from the underlying Dataset. This `tf.data.Dataset` will load and collate batches from
        the Dataset, and is suitable for passing to methods like `model.fit()` or `model.predict()`. The dataset will yield
        `dicts` for both inputs and labels unless the `dict` would contain only a single key, in which case a raw
        `tf.Tensor` is yielded instead.

        Args:
            batch_size (`int`, *optional*):
                Size of batches to load from the dataset. Defaults to `None`, which implies that the dataset won't be
                batched, but the returned dataset can be batched later with `tf_dataset.batch(batch_size)`.
            columns (`List[str]` or `str`, *optional*):
                Dataset column(s) to load in the `tf.data.Dataset`.
                Column names that are created by the `collate_fn` and that do not exist in the original dataset can be used.
            shuffle(`bool`, defaults to `False`):
                Shuffle the dataset order when loading. Recommended `True` for training, `False` for
                validation/evaluation.
            drop_remainder(`bool`, defaults to `False`):
                Drop the last incomplete batch when loading. Ensures
                that all batches yielded by the dataset will have the same length on the batch dimension.
            collate_fn(`Callable`, *optional*):
                A function or callable object (such as a `DataCollator`) that will collate
                lists of samples into a batch.
            collate_fn_args (`Dict`, *optional*):
                An optional `dict` of keyword arguments to be passed to the
                `collate_fn`.
            label_cols (`List[str]` or `str`, defaults to `None`):
                Dataset column(s) to load as labels.
                Note that many models compute loss internally rather than letting Keras do it, in which case
                passing the labels here is optional, as long as they're in the input `columns`.
            prefetch (`bool`, defaults to `True`):
                Whether to run the dataloader in a separate thread and maintain
                a small buffer of batches for training. Improves performance by allowing data to be loaded in the
                background while the model is training.
            num_workers (`int`, defaults to `0`):
                Number of workers to use for loading the dataset. Only supported on Python versions >= 3.8.
            num_test_batches (`int`, defaults to `20`):
                Number of batches to use to infer the output signature of the dataset.
                The higher this number, the more accurate the signature will be, but the longer it will take to
                create the dataset.

        Returns:
            `tf.data.Dataset`

        Example:

        ```py
        >>> ds_train = ds["train"].to_tf_dataset(
        ...    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
        ...    shuffle=True,
        ...    batch_size=16,
        ...    collate_fn=data_collator,
        ... )
        ```
        """
        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError('Called a Tensorflow-specific function but Tensorflow is not installed.')
        if isinstance(columns, list) and len(columns) == 1 or (isinstance(label_cols, list) and len(label_cols) == 1):
            warnings.warn("The output of `to_tf_dataset` will change when a passing single element list for `labels` or `columns` in the next datasets version. To return a tuple structure rather than dict, pass a single string.\nOld behaviour: columns=['a'], labels=['labels'] -> (tf.Tensor, tf.Tensor)  \n             : columns='a', labels='labels' -> (tf.Tensor, tf.Tensor)  \nNew behaviour: columns=['a'],labels=['labels'] -> ({'a': tf.Tensor}, {'labels': tf.Tensor})  \n             : columns='a', labels='labels' -> (tf.Tensor, tf.Tensor) ", FutureWarning)
        if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
            logger.warning('Note that to_tf_dataset() loads the data with a generator rather than a full tf.data pipeline and is not compatible with remote TPU connections. If you encounter errors, please try using a TPU VM or, if your data can fit in memory, loading it into memory as a dict of Tensors instead of streaming with to_tf_dataset().')
        if collate_fn is None:
            collate_fn = minimal_tf_collate_fn
        if collate_fn_args is None:
            collate_fn_args = {}
        if label_cols and (not columns):
            raise ValueError('Cannot specify label_cols without specifying columns!')
        if label_cols is None:
            label_cols = []
        elif isinstance(label_cols, str):
            label_cols = [label_cols]
        if len(set(label_cols)) < len(label_cols):
            raise ValueError('List of label_cols contains duplicates.')
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            if len(set(columns)) < len(columns):
                raise ValueError('List of columns contains duplicates.')
            cols_to_retain = list(set(columns + label_cols))
        else:
            cols_to_retain = None
            columns = []
        if self.format['type'] not in ['custom', 'numpy']:
            dataset = self.with_format('numpy')
        else:
            dataset = self
        output_signature, columns_to_np_types = dataset._get_output_signature(dataset, collate_fn=collate_fn, collate_fn_args=collate_fn_args, cols_to_retain=cols_to_retain, batch_size=batch_size if drop_remainder else None, num_test_batches=num_test_batches)
        if 'labels' in output_signature:
            if ('label_ids' in columns or 'label' in columns) and 'labels' not in columns:
                columns = [col for col in columns if col not in ['label_ids', 'label']] + ['labels']
            if ('label_ids' in label_cols or 'label' in label_cols) and 'labels' not in label_cols:
                label_cols = [col for col in label_cols if col not in ['label_ids', 'label']] + ['labels']
        for col in columns:
            if col not in output_signature:
                raise ValueError(f'Column {col} not found in dataset!')
        for col in label_cols:
            if col not in output_signature:
                raise ValueError(f'Label column {col} not found in dataset!')
        if num_workers == 0:
            tf_dataset = dataset_to_tf(dataset=dataset, cols_to_retain=cols_to_retain, collate_fn=collate_fn, collate_fn_args=collate_fn_args, columns_to_np_types=columns_to_np_types, output_signature=output_signature, shuffle=shuffle, batch_size=batch_size, drop_remainder=drop_remainder)
        elif num_workers > 0:
            if batch_size is None:
                raise NotImplementedError('`batch_size` must be specified when using multiple workers, as unbatched multiprocessing is not supported yet. Please provide a `batch_size` if `num_workers` is greater than 0.')
            tf_dataset = multiprocess_dataset_to_tf(dataset=dataset, cols_to_retain=cols_to_retain, collate_fn=collate_fn, collate_fn_args=collate_fn_args, columns_to_np_types=columns_to_np_types, output_signature=output_signature, shuffle=shuffle, batch_size=batch_size, drop_remainder=drop_remainder, num_workers=num_workers)
        else:
            raise ValueError('num_workers must be >= 0')

        def split_features_and_labels(input_batch):
            features = {key: tensor for key, tensor in input_batch.items() if key in columns}
            labels = {key: tensor for key, tensor in input_batch.items() if key in label_cols}
            if len(features) == 1:
                features = list(features.values())[0]
            if len(labels) == 1:
                labels = list(labels.values())[0]
            if isinstance(labels, dict) and len(labels) == 0:
                return features
            else:
                return (features, labels)
        if cols_to_retain is not None:
            tf_dataset = tf_dataset.map(split_features_and_labels)
        if prefetch:
            tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        def cleanup_callback(ref):
            dataset.__del__()
            self._TF_DATASET_REFS.remove(ref)
        self._TF_DATASET_REFS.add(weakref.ref(tf_dataset, cleanup_callback))
        return tf_dataset