from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from fsspec.core import expand_paths_if_needed, get_fs_token_paths, stringify_path
from fsspec.spec import AbstractFileSystem
from dask import config
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key, parse_bytes
def _aggregate_stats(file_path, file_row_group_stats, file_row_group_column_stats, stat_col_indices):
    """Utility to aggregate the statistics for N row-groups
    into a single dictionary.

    Used by `Engine._construct_parts`
    """
    if len(file_row_group_stats) < 1:
        return {}
    elif len(file_row_group_column_stats) == 0:
        assert len(file_row_group_stats) == 1
        return file_row_group_stats[0]
    else:
        if len(file_row_group_stats) > 1:
            df_rgs = pd.DataFrame(file_row_group_stats)
            s = {'file_path_0': file_path, 'num-rows': df_rgs['num-rows'].sum(), 'num-row-groups': df_rgs['num-rows'].count(), 'total_byte_size': df_rgs['total_byte_size'].sum(), 'columns': []}
        else:
            s = {'file_path_0': file_path, 'num-rows': file_row_group_stats[0]['num-rows'], 'num-row-groups': 1, 'total_byte_size': file_row_group_stats[0]['total_byte_size'], 'columns': []}
        df_cols = None
        if len(file_row_group_column_stats) > 1:
            df_cols = pd.DataFrame(file_row_group_column_stats)
        for ind, name in enumerate(stat_col_indices):
            i = ind * 3
            if df_cols is None:
                minval = file_row_group_column_stats[0][i]
                maxval = file_row_group_column_stats[0][i + 1]
                null_count = file_row_group_column_stats[0][i + 2]
                if minval == maxval and null_count:
                    s['columns'].append({'null_count': null_count})
                else:
                    s['columns'].append({'name': name, 'min': minval, 'max': maxval, 'null_count': null_count})
            else:
                minval = df_cols.iloc[:, i].dropna().min()
                maxval = df_cols.iloc[:, i + 1].dropna().max()
                null_count = df_cols.iloc[:, i + 2].sum()
                if minval == maxval and null_count:
                    s['columns'].append({'null_count': null_count})
                else:
                    s['columns'].append({'name': name, 'min': minval, 'max': maxval, 'null_count': null_count})
        return s