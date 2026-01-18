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
class FileMetadataProvider:
    """Abstract callable that provides metadata for the files of a single dataset block.

    Current subclasses:
        - :class:`BaseFileMetadataProvider`
        - :class:`ParquetMetadataProvider`
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], **kwargs) -> BlockMetadata:
        """Resolves and returns block metadata for files in the given paths.

        All file paths provided should belong to a single dataset block.

        Args:
            paths: The file paths for a single dataset block.
            schema: The user-provided or inferred schema for the given paths,
                if any.

        Returns:
            BlockMetadata aggregated across the given paths.
        """
        raise NotImplementedError

    def __call__(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], **kwargs) -> BlockMetadata:
        return self._get_block_metadata(paths, schema, **kwargs)