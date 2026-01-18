import logging
import threading
from contextlib import nullcontext
from typing import Any, Callable, Iterator, List, Optional, Tuple
import ray
from ray.actor import ActorHandle
from ray.data._internal.batcher import Batcher, ShufflingBatcher
from ray.data._internal.block_batching.interfaces import (
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _calculate_ref_hits(refs: List[ObjectRef[Any]]) -> Tuple[int, int, int]:
    """Given a list of object references, returns how many are already on the local
    node, how many require fetching from another node, and how many have unknown
    locations."""
    current_node_id = ray.get_runtime_context().get_node_id()
    ctx = ray.data.context.DataContext.get_current()
    if ctx.enable_get_object_locations_for_metrics:
        locs = ray.experimental.get_object_locations(refs)
    else:
        locs = {}
    nodes: List[List[str]] = [loc['node_ids'] for loc in locs.values()]
    hits = sum((current_node_id in node_ids for node_ids in nodes))
    unknowns = sum((1 for node_ids in nodes if not node_ids))
    misses = len(nodes) - hits - unknowns
    return (hits, misses, unknowns)