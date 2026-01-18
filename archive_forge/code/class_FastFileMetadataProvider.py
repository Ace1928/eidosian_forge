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
class FastFileMetadataProvider(DefaultFileMetadataProvider):
    """Fast Metadata provider for
    :class:`~ray.data.datasource.file_based_datasource.FileBasedDatasource`
    implementations.

    Offers improved performance vs.
    :class:`DefaultFileMetadataProvider`
    by skipping directory path expansion and file size collection.
    While this performance improvement may be negligible for local filesystems,
    it can be substantial for cloud storage service providers.

    This should only be used when all input paths exist and are known to be files.
    """

    def expand_paths(self, paths: List[str], filesystem: 'pyarrow.fs.FileSystem', partitioning: Optional[Partitioning]=None, ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
        if ignore_missing_paths:
            raise ValueError('`ignore_missing_paths` cannot be set when used with `FastFileMetadataProvider`. All paths must exist when using `FastFileMetadataProvider`.')
        logger.warning(f'Skipping expansion of {len(paths)} path(s). If your paths contain directories or if file size collection is required, try rerunning this read with `meta_provider=DefaultFileMetadataProvider()`.')
        yield from zip(paths, itertools.repeat(None, len(paths)))