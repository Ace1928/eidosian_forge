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
def _get_file_infos_serial(paths: List[str], filesystem: 'pyarrow.fs.FileSystem', ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    for path in paths:
        yield from _get_file_infos(path, filesystem, ignore_missing_paths)