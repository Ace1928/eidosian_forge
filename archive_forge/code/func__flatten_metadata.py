import math
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats, _get_or_create_stats_actor
from ray.data._internal.util import _split_list
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def _flatten_metadata(self, metadata: List[BlockPartitionMetadata]) -> List[BlockMetadata]:
    """Flatten the metadata of computed blocks into a list.

        This is required because dynamic block splitting can produce multiple output
        blocks from each task.
        """
    return [meta for meta_list in metadata for meta in meta_list]