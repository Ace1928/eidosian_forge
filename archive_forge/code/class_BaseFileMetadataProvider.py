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
class BaseFileMetadataProvider(FileMetadataProvider):
    """Abstract callable that provides metadata for
    :class:`~ray.data.datasource.file_based_datasource.FileBasedDatasource`
    implementations that reuse the base :meth:`~ray.data.Datasource.prepare_read`
    method.

    Also supports file and file size discovery in input directory paths.

    Current subclasses:
        - :class:`DefaultFileMetadataProvider`
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, rows_per_file: Optional[int], file_sizes: List[Optional[int]]) -> BlockMetadata:
        """Resolves and returns block metadata for files of a single dataset block.

        Args:
            paths: The file paths for a single dataset block. These
                paths will always be a subset of those previously returned from
                :meth:`.expand_paths`.
            schema: The user-provided or inferred schema for the given file
                paths, if any.
            rows_per_file: The fixed number of rows per input file, or None.
            file_sizes: Optional file size per input file previously returned
                from :meth:`.expand_paths`, where `file_sizes[i]` holds the size of
                the file at `paths[i]`.

        Returns:
            BlockMetadata aggregated across the given file paths.
        """
        raise NotImplementedError

    def expand_paths(self, paths: List[str], filesystem: Optional['pyarrow.fs.FileSystem'], partitioning: Optional[Partitioning]=None, ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
        """Expands all paths into concrete file paths by walking directories.

        Also returns a sidecar of file sizes.

        The input paths must be normalized for compatibility with the input
        filesystem prior to invocation.

        Args:
            paths: A list of file and/or directory paths compatible with the
                given filesystem.
            filesystem: The filesystem implementation that should be used for
                expanding all paths and reading their files.
            ignore_missing_paths: If True, ignores any file paths in ``paths`` that
                are not found. Defaults to False.

        Returns:
            An iterator of `(file_path, file_size)` pairs. None may be returned for the
            file size if it is either unknown or will be fetched later by
            `_get_block_metadata()`, but the length of
            both lists must be equal.
        """
        raise NotImplementedError