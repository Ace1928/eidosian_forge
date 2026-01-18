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
class ParquetMetadataProvider(FileMetadataProvider):
    """Abstract callable that provides block metadata for Arrow Parquet file fragments.

    All file fragments should belong to a single dataset block.

    Supports optional pre-fetching of ordered metadata for all file fragments in
    a single batch to help optimize metadata resolution.

    Current subclasses:
        - :class:`~ray.data.datasource.file_meta_provider.DefaultParquetMetadataProvider`
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, num_fragments: int, prefetched_metadata: Optional[List[Any]]) -> BlockMetadata:
        """Resolves and returns block metadata for files of a single dataset block.

        Args:
            paths: The file paths for a single dataset block.
            schema: The user-provided or inferred schema for the given file
                paths, if any.
            num_fragments: The number of Parquet file fragments derived from the input
                file paths.
            prefetched_metadata: Metadata previously returned from
                `prefetch_file_metadata()` for each file fragment, where
                `prefetched_metadata[i]` contains the metadata for `fragments[i]`.

        Returns:
            BlockMetadata aggregated across the given file paths.
        """
        raise NotImplementedError

    def prefetch_file_metadata(self, fragments: List['pyarrow.dataset.ParquetFileFragment'], **ray_remote_args) -> Optional[List[Any]]:
        """Pre-fetches file metadata for all Parquet file fragments in a single batch.

        Subsets of the metadata returned will be provided as input to
        subsequent calls to :meth:`~FileMetadataProvider._get_block_metadata` together
        with their corresponding Parquet file fragments.

        Implementations that don't support pre-fetching file metadata shouldn't
        override this method.

        Args:
            fragments: The Parquet file fragments to fetch metadata for.

        Returns:
            Metadata resolved for each input file fragment, or `None`. Metadata
            must be returned in the same order as all input file fragments, such
            that `metadata[i]` always contains the metadata for `fragments[i]`.
        """
        return None