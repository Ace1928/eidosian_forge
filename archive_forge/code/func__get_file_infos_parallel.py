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
def _get_file_infos_parallel(paths: List[str], filesystem: 'pyarrow.fs.FileSystem', ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    from ray.data.datasource.file_based_datasource import PATHS_PER_FILE_SIZE_FETCH_TASK, _unwrap_s3_serialization_workaround, _wrap_s3_serialization_workaround
    logger.warning(f'Expanding {len(paths)} path(s). This may be a HIGH LATENCY operation on some cloud storage services. Moving all the paths to a common parent directory will lead to faster metadata fetching.')
    filesystem = _wrap_s3_serialization_workaround(filesystem)

    def _file_infos_fetcher(paths: List[str]) -> List[Tuple[str, int]]:
        fs = _unwrap_s3_serialization_workaround(filesystem)
        return list(itertools.chain.from_iterable((_get_file_infos(path, fs, ignore_missing_paths) for path in paths)))
    yield from _fetch_metadata_parallel(paths, _file_infos_fetcher, PATHS_PER_FILE_SIZE_FETCH_TASK)