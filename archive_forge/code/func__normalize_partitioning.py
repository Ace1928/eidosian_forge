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
def _normalize_partitioning(cls, remote_parts, row_lengths, column_widths):
    """
        Normalize partitioning according to the default partitioning scheme in Modin.

        The result of 'read_parquet()' is often under partitioned over rows and over partitioned
        over columns, so this method expands the number of row splits and shrink the number of column splits.

        Parameters
        ----------
        remote_parts : np.ndarray
        row_lengths : list of ints or None
            Row lengths, if 'None', won't repartition across rows.
        column_widths : list of ints

        Returns
        -------
        remote_parts : np.ndarray
        row_lengths : list of ints or None
        column_widths : list of ints
        """
    if len(remote_parts) == 0:
        return (remote_parts, row_lengths, column_widths)
    from modin.core.storage_formats.pandas.utils import get_length_list
    actual_row_nparts = remote_parts.shape[0]
    if row_lengths is not None:
        desired_row_nparts = max(1, min(sum(row_lengths) // MinPartitionSize.get(), NPartitions.get()))
    else:
        desired_row_nparts = actual_row_nparts
    if 1.5 * actual_row_nparts < desired_row_nparts:
        splits_per_partition = desired_row_nparts // actual_row_nparts
        remainder = desired_row_nparts % actual_row_nparts
        new_parts = []
        new_row_lengths = []
        for row_idx, (part_len, row_parts) in enumerate(zip(row_lengths, remote_parts)):
            num_splits = splits_per_partition
            if row_idx < remainder:
                num_splits += 1
            if num_splits == 1:
                new_parts.append(row_parts)
                new_row_lengths.append(part_len)
                continue
            offset = len(new_parts)
            new_parts.extend([[] for _ in range(num_splits)])
            for part in row_parts:
                split = cls.frame_cls._partition_mgr_cls._column_partitions_class([part]).apply(lambda df: df, num_splits=num_splits, maintain_partitioning=False)
                for i in range(num_splits):
                    new_parts[offset + i].append(split[i])
            new_row_lengths.extend(get_length_list(part_len, num_splits, MinPartitionSize.get()))
        remote_parts = np.array(new_parts)
        row_lengths = new_row_lengths
    desired_col_nparts = max(1, min(sum(column_widths) // MinPartitionSize.get(), NPartitions.get()))
    if 1.5 * desired_col_nparts < remote_parts.shape[1]:
        remote_parts = np.array([cls.frame_cls._partition_mgr_cls._row_partition_class(row_parts).apply(lambda df: df, num_splits=desired_col_nparts, maintain_partitioning=False) for row_parts in remote_parts])
        column_widths = get_length_list(sum(column_widths), desired_col_nparts, MinPartitionSize.get())
    return (remote_parts, row_lengths, column_widths)