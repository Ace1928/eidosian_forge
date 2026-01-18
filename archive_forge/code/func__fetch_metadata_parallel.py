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
def _fetch_metadata_parallel(uris: List[Uri], fetch_func: Callable[[List[Uri]], List[Meta]], desired_uris_per_task: int, **ray_remote_args) -> Iterator[Meta]:
    """Fetch file metadata in parallel using Ray tasks."""
    remote_fetch_func = cached_remote_fn(fetch_func, num_cpus=0.5)
    if ray_remote_args:
        remote_fetch_func = remote_fetch_func.options(**ray_remote_args)
    parallelism = max(len(uris) // desired_uris_per_task, 2)
    metadata_fetch_bar = ProgressBar('Metadata Fetch Progress', total=parallelism)
    fetch_tasks = []
    for uri_chunk in np.array_split(uris, parallelism):
        if len(uri_chunk) == 0:
            continue
        fetch_tasks.append(remote_fetch_func.remote(uri_chunk))
    results = metadata_fetch_bar.fetch_until_complete(fetch_tasks)
    yield from itertools.chain.from_iterable(results)