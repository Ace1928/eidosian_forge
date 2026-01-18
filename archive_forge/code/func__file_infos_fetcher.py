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
def _file_infos_fetcher(paths: List[str]) -> List[Tuple[str, int]]:
    fs = _unwrap_s3_serialization_workaround(filesystem)
    return list(itertools.chain.from_iterable((_get_file_infos(path, fs, ignore_missing_paths) for path in paths)))