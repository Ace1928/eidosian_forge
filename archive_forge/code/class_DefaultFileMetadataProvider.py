import itertools
import logging
import os
import pathlib
import re
from typing import (
import numpy as np
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class DefaultFileMetadataProvider(BaseFileMetadataProvider):
    """Default metadata provider for
    :class:`~ray.data.datasource.file_based_datasource.FileBasedDatasource`
    implementations that reuse the base `prepare_read` method.

    Calculates block size in bytes as the sum of its constituent file sizes,
    and assumes a fixed number of rows per file.
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, rows_per_file: Optional[int], file_sizes: List[Optional[int]]) -> BlockMetadata:
        if rows_per_file is None:
            num_rows = None
        else:
            num_rows = len(paths) * rows_per_file
        return BlockMetadata(num_rows=num_rows, size_bytes=None if None in file_sizes else int(sum(file_sizes)), schema=schema, input_files=paths, exec_stats=None)

    def expand_paths(self, paths: List[str], filesystem: 'pyarrow.fs.FileSystem', partitioning: Optional[Partitioning]=None, ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
        yield from _expand_paths(paths, filesystem, partitioning, ignore_missing_paths)