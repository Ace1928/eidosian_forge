import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def _autodetect_parallelism(parallelism: int, target_max_block_size: int, ctx: DataContext, datasource_or_legacy_reader: Optional[Union['Datasource', 'Reader']]=None, mem_size: Optional[int]=None, placement_group: Optional['PlacementGroup']=None, avail_cpus: Optional[int]=None) -> (int, str, int, Optional[int]):
    """Returns parallelism to use and the min safe parallelism to avoid OOMs.

    This detects parallelism using the following heuristics, applied in order:

     1) We start with the default parallelism of 200. This can be overridden by
        setting the `min_parallelism` attribute of
        :class:`~ray.data.context.DataContext`.
     2) Min block size. If the parallelism would make blocks smaller than this
        threshold, the parallelism is reduced to avoid the overhead of tiny blocks.
     3) Max block size. If the parallelism would make blocks larger than this
        threshold, the parallelism is increased to avoid OOMs during processing.
     4) Available CPUs. If the parallelism cannot make use of all the available
        CPUs in the cluster, the parallelism is increased until it can.

    Args:
        parallelism: The user-requested parallelism, or -1 for auto-detection.
        target_max_block_size: The target max block size to
            produce. We pass this separately from the
            DatasetContext because it may be set per-op instead of
            per-Dataset.
        ctx: The current Dataset context to use for configs.
        datasource_or_legacy_reader: The datasource or legacy reader, to be used for
            data size estimation.
        mem_size: If passed, then used to compute the parallelism according to
            target_max_block_size.
        placement_group: The placement group that this Dataset
            will execute inside, if any.
        avail_cpus: Override avail cpus detection (for testing only).

    Returns:
        Tuple of detected parallelism (only if -1 was specified), the reason
        for the detected parallelism (only if -1 was specified), the min safe
        parallelism (which can be used to generate warnings about large
        blocks), and the estimated inmemory size of the dataset.
    """
    min_safe_parallelism = 1
    max_reasonable_parallelism = sys.maxsize
    if mem_size is None and datasource_or_legacy_reader:
        mem_size = datasource_or_legacy_reader.estimate_inmemory_data_size()
    if mem_size is not None and (not np.isnan(mem_size)):
        min_safe_parallelism = max(1, int(mem_size / target_max_block_size))
        max_reasonable_parallelism = max(1, int(mem_size / ctx.target_min_block_size))
    reason = ''
    if parallelism < 0:
        if parallelism != -1:
            raise ValueError('`parallelism` must either be -1 or a positive integer.')
        if placement_group is None:
            placement_group = ray.util.get_current_placement_group()
        avail_cpus = avail_cpus or _estimate_avail_cpus(placement_group)
        parallelism = max(min(ctx.min_parallelism, max_reasonable_parallelism), min_safe_parallelism, avail_cpus * 2)
        if parallelism == ctx.min_parallelism:
            reason = f'DataContext.get_current().min_parallelism={ctx.min_parallelism}'
        elif parallelism == max_reasonable_parallelism:
            reason = f'output blocks of size at least DataContext.get_current().target_min_block_size={ctx.target_min_block_size / (1024 * 1024)}MiB'
        elif parallelism == min_safe_parallelism:
            reason = f'output blocks of size at most DataContext.get_current().target_max_block_size={ctx.target_max_block_size / (1024 * 1024)}MiB'
        else:
            reason = f'parallelism at least twice the available number of CPUs ({avail_cpus})'
        logger.debug(f'Autodetected parallelism={parallelism} based on estimated_available_cpus={avail_cpus} and estimated_data_size={mem_size}.')
    return (parallelism, reason, min_safe_parallelism, mem_size)