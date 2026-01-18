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
def _concatenate_map_style_datasets(dsets: List[Dataset], info: Optional[DatasetInfo]=None, split: Optional[NamedSplit]=None, axis: int=0):
    """
    Converts a list of :class:`Dataset` with the same schema into a single :class:`Dataset`.
    When you concatenate on axis 0, missing data are filled with None values.

    Args:
        dsets (`List[datasets.Dataset]`): List of Datasets to concatenate.
        info (:class:`DatasetInfo`, optional): Dataset information, like description, citation, etc.
        split (:class:`NamedSplit`, optional): Name of the dataset split.
        axis (``{0, 1}``, default ``0``, meaning over rows):
            Axis to concatenate over, where ``0`` means over rows (vertically) and ``1`` means over columns
            (horizontally).

            *New in version 1.6.0*

    Example:

    ```py
    >>> ds3 = _concatenate_map_style_datasets([ds1, ds2])
    ```
    """
    if any((dset.num_rows > 0 for dset in dsets)):
        dsets = [dset for dset in dsets if dset.num_rows > 0]
    else:
        return dsets[0]
    if axis == 0:
        _check_if_features_can_be_aligned([dset.features for dset in dsets])
    else:
        if not all((dset.num_rows == dsets[0].num_rows for dset in dsets)):
            raise ValueError('Number of rows must match for all datasets')
        _check_column_names([col_name for dset in dsets for col_name in dset._data.column_names])
    format = dsets[0].format
    if any((dset.format != format for dset in dsets)):
        format = {}
        logger.info('Some of the datasets have disparate format. Resetting the format of the concatenated dataset.')

    def apply_offset_to_indices_table(table, offset):
        if offset == 0:
            return table
        else:
            array = table['indices']
            new_array = pc.add(array, pa.scalar(offset, type=pa.uint64()))
            return InMemoryTable.from_arrays([new_array], names=['indices'])
    if any((dset._indices is not None for dset in dsets)):
        if axis == 0:
            indices_tables = []
            for i in range(len(dsets)):
                if dsets[i]._indices is None:
                    dsets[i] = dsets[i]._select_with_indices_mapping(range(len(dsets[i])))
                indices_tables.append(dsets[i]._indices)
            offset = 0
            for i in range(len(dsets)):
                indices_tables[i] = apply_offset_to_indices_table(indices_tables[i], offset)
                offset += len(dsets[i]._data)
            indices_tables = [t for t in indices_tables if len(t) > 0]
            if indices_tables:
                indices_table = concat_tables(indices_tables)
            else:
                indices_table = InMemoryTable.from_batches([], schema=pa.schema({'indices': pa.int64()}))
        elif len(dsets) == 1:
            indices_table = dsets[0]._indices
        else:
            for i in range(len(dsets)):
                dsets[i] = dsets[i].flatten_indices()
            indices_table = None
    else:
        indices_table = None
    table = concat_tables([dset._data for dset in dsets], axis=axis)
    if axis == 0:
        features_list = _align_features([dset.features for dset in dsets])
    else:
        features_list = [dset.features for dset in dsets]
    table = update_metadata_with_features(table, {k: v for features in features_list for k, v in features.items()})
    if info is None:
        info = DatasetInfo.from_merge([dset.info for dset in dsets])
    fingerprint = update_fingerprint(''.join((dset._fingerprint for dset in dsets)), _concatenate_map_style_datasets, {'info': info, 'split': split})
    concatenated_dataset = Dataset(table, info=info, split=split, indices_table=indices_table, fingerprint=fingerprint)
    concatenated_dataset.set_format(**format)
    return concatenated_dataset