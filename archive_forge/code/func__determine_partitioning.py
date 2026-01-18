import json
import os
import re
from typing import TYPE_CHECKING
import fsspec
import numpy as np
import pandas
import pandas._libs.lib as lib
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
from packaging import version
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
@classmethod
def _determine_partitioning(cls, dataset: ColumnStoreDataset) -> 'list[list[ParquetFileToRead]]':
    """
        Determine which partition will read certain files/row groups of the dataset.

        Parameters
        ----------
        dataset : ColumnStoreDataset

        Returns
        -------
        list[list[ParquetFileToRead]]
            Each element in the returned list describes a list of files that a partition has to read.
        """
    from modin.core.storage_formats.pandas.parsers import ParquetFileToRead
    parquet_files = dataset.files
    row_groups_per_file = dataset.row_groups_per_file
    num_row_groups = sum(row_groups_per_file)
    if num_row_groups == 0:
        return []
    num_splits = min(NPartitions.get(), num_row_groups)
    part_size = num_row_groups // num_splits
    reminder = num_row_groups % num_splits
    part_sizes = [part_size] * (num_splits - reminder) + [part_size + 1] * reminder
    partition_files = []
    file_idx = 0
    row_group_idx = 0
    row_groups_left_in_current_file = row_groups_per_file[file_idx]
    total_row_groups_added = 0
    for size in part_sizes:
        row_groups_taken = 0
        part_files = []
        while row_groups_taken != size:
            if row_groups_left_in_current_file < 1:
                file_idx += 1
                row_group_idx = 0
                row_groups_left_in_current_file = row_groups_per_file[file_idx]
            to_take = min(size - row_groups_taken, row_groups_left_in_current_file)
            part_files.append(ParquetFileToRead(parquet_files[file_idx], row_group_start=row_group_idx, row_group_end=row_group_idx + to_take))
            row_groups_left_in_current_file -= to_take
            row_groups_taken += to_take
            row_group_idx += to_take
        total_row_groups_added += row_groups_taken
        partition_files.append(part_files)
    sanity_check = len(partition_files) == num_splits and total_row_groups_added == num_row_groups
    ErrorMessage.catch_bugs_and_request_email(failure_condition=not sanity_check, extra_log='row groups added does not match total num of row groups across parquet files')
    return partition_files