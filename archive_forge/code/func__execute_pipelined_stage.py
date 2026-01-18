import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _execute_pipelined_stage(stage_fn: Callable[[T], Tuple[ObjectRef, U]], prev_metadata_refs: List[ObjectRef], stage_args: List[T], progress_bar: Optional[ProgressBar]=None) -> Tuple[List[BlockMetadata], List[ObjectRef], List[U]]:
    """
    Helper function to execute a stage of tasks. This will wait for the
    previous round of tasks to complete before submitting the next.
    """
    if progress_bar is not None:
        prev_metadata = progress_bar.fetch_until_complete(prev_metadata_refs)
    else:
        prev_metadata = ray.get(prev_metadata_refs)
    prev_metadata_refs.clear()
    metadata_refs = []
    data_outputs = []
    while stage_args:
        arg = stage_args.pop(0)
        metadata_ref, data_output = stage_fn(arg)
        metadata_refs.append(metadata_ref)
        data_outputs.append(data_output)
    return (prev_metadata, metadata_refs, data_outputs)