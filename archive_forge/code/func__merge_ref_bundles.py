import copy
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Union
import ray
from ray import ObjectRef
from ray._raylet import ObjectRefGenerator
from ray.data._internal.compute import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _merge_ref_bundles(*bundles: RefBundle) -> RefBundle:
    """Merge N ref bundles into a single bundle of multiple blocks."""
    assert any((bundle is not None for bundle in bundles))
    blocks = list(itertools.chain((block for bundle in bundles if bundle is not None for block in bundle.blocks)))
    owns_blocks = all((bundle.owns_blocks for bundle in bundles if bundle is not None))
    return RefBundle(blocks, owns_blocks)