from __future__ import annotations
import copy
import pickle
import threading
import warnings
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
import numpy as np
import pandas as pd
import tlz as toolz
from packaging.version import parse as parse_version
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_201
from dask.base import tokenize
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _is_local_fs, _meta_from_dtypes, _open_input_files
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.delayed import Delayed
from dask.utils import natural_sort_key
@classmethod
def _construct_collection_plan(cls, dataset_info):
    fs = dataset_info['fs']
    parts = dataset_info['parts']
    paths = dataset_info['paths']
    filters = dataset_info['filters']
    pf = dataset_info['pf']
    split_row_groups = dataset_info['split_row_groups']
    blocksize = dataset_info['blocksize']
    gather_statistics = dataset_info['gather_statistics']
    base_path = dataset_info['base']
    aggregation_depth = dataset_info['aggregation_depth']
    index_cols = dataset_info['index_cols']
    categories = dataset_info['categories']
    dtypes = dataset_info['dtypes']
    categories_dict = dataset_info['categories_dict']
    has_metadata_file = dataset_info['has_metadata_file']
    metadata_task_size = dataset_info['metadata_task_size']
    kwargs = dataset_info['kwargs']
    metadata_task_size = _set_metadata_task_size(dataset_info['metadata_task_size'], fs)
    filter_columns = {t[0] for t in flatten(filters or [], container=list)}
    stat_col_indices = {}
    _index_cols = index_cols if gather_statistics and len(index_cols) == 1 else []
    for i, name in enumerate(pf.columns):
        if name in _index_cols or name in filter_columns:
            stat_col_indices[name] = i
    gather_statistics = _set_gather_statistics(gather_statistics, blocksize, split_row_groups, aggregation_depth, filter_columns, set(stat_col_indices) | filter_columns)
    common_kwargs = {'categories': categories_dict or categories, 'root_cats': pf.cats, 'root_file_scheme': pf.file_scheme, 'base_path': base_path, **kwargs}
    if gather_statistics is False and (not split_row_groups) and isinstance(parts, list) and len(parts) and isinstance(parts[0], str):
        return ([{'piece': (full_path, None)} for full_path in parts], [], common_kwargs)
    dataset_info_kwargs = {'fs': fs, 'split_row_groups': split_row_groups, 'gather_statistics': gather_statistics, 'filters': filters, 'dtypes': dtypes, 'stat_col_indices': stat_col_indices, 'aggregation_depth': aggregation_depth, 'blocksize': blocksize, 'root_cats': pf.cats, 'root_file_scheme': pf.file_scheme, 'base_path': '' if base_path is None else base_path, 'has_metadata_file': has_metadata_file}
    if has_metadata_file or metadata_task_size == 0 or metadata_task_size > len(paths):
        pf_or_paths = pf if has_metadata_file else paths
        parts, stats = cls._collect_file_parts(pf_or_paths, dataset_info_kwargs)
    else:
        parts, stats = ([], [])
        if paths:
            gather_parts_dsk = {}
            name = 'gather-pq-parts-' + tokenize(paths, dataset_info_kwargs)
            finalize_list = []
            for task_i, file_i in enumerate(range(0, len(paths), metadata_task_size)):
                finalize_list.append((name, task_i))
                gather_parts_dsk[finalize_list[-1]] = (cls._collect_file_parts, paths[file_i:file_i + metadata_task_size], dataset_info_kwargs)

            def _combine_parts(parts_and_stats):
                parts, stats = ([], [])
                for part, stat in parts_and_stats:
                    parts += part
                    if stat:
                        stats += stat
                return (parts, stats)
            gather_parts_dsk['final-' + name] = (_combine_parts, finalize_list)
            parts, stats = Delayed('final-' + name, gather_parts_dsk).compute()
    return (parts, stats, common_kwargs)