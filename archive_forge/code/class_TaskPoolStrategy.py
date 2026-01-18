import collections
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import (
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class TaskPoolStrategy(ComputeStrategy):

    def _apply(self, block_fn: BlockTransform, remote_args: dict, block_list: BlockList, clear_input_blocks: bool, name: Optional[str]=None, min_rows_per_block: Optional[int]=None, fn: Optional[UserDefinedFunction]=None, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None) -> BlockList:
        assert not DataContext.get_current().new_execution_backend, 'Legacy backend off'
        assert fn_constructor_args is None and fn_constructor_kwargs is None
        if fn_args is None:
            fn_args = tuple()
        if fn_kwargs is None:
            fn_kwargs = {}
        if block_list.initial_num_blocks() == 0:
            return block_list
        if name is None:
            name = 'map'
        blocks = block_list.get_blocks_with_metadata()
        if min_rows_per_block is not None:
            _check_batch_size(blocks, min_rows_per_block, name)
            block_bundles = _bundle_blocks_up_to_size(blocks, min_rows_per_block)
        else:
            block_bundles = [((b,), (m,)) for b, m in blocks]
        del blocks
        name = name.title()
        map_bar = ProgressBar(name, total=len(block_bundles))
        map_block = cached_remote_fn(_map_block_split).options(num_returns='dynamic', **remote_args)
        refs = [map_block.remote(block_fn, [f for m in ms for f in m.input_files], fn, len(bs), *bs + fn_args, **fn_kwargs) for bs, ms in block_bundles]
        in_block_owned_by_consumer = block_list._owned_by_consumer
        if clear_input_blocks:
            del block_bundles
            block_list.clear()
        try:
            results = map_bar.fetch_until_complete(refs)
        except (ray.exceptions.RayError, KeyboardInterrupt) as e:
            for ref in refs:
                ray.cancel(ref)
            for ref in refs:
                try:
                    ray.get(ref)
                except ray.exceptions.RayError:
                    pass
            raise e from None
        new_blocks, new_metadata = ([], [])
        for ref_generator in results:
            refs = list(ref_generator)
            metadata = ray.get(refs.pop(-1))
            assert len(metadata) == len(refs)
            new_blocks += refs
            new_metadata += metadata
        return BlockList(list(new_blocks), list(new_metadata), owned_by_consumer=in_block_owned_by_consumer)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TaskPoolStrategy) or other == 'tasks'