from typing import Any, Dict, List, Optional, Tuple, Union
from ray.data._internal.block_list import BlockList
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import Block, BlockMetadata
class SimpleShufflePlan(ShuffleOp):

    def execute(self, input_blocks: BlockList, output_num_blocks: int, clear_input_blocks: bool, *, map_ray_remote_args: Optional[Dict[str, Any]]=None, reduce_ray_remote_args: Optional[Dict[str, Any]]=None, ctx: Optional[TaskContext]=None) -> Tuple[BlockList, Dict[str, List[BlockMetadata]]]:
        input_blocks_list = input_blocks.get_blocks()
        input_num_blocks = len(input_blocks_list)
        if map_ray_remote_args is None:
            map_ray_remote_args = {}
        if reduce_ray_remote_args is None:
            reduce_ray_remote_args = {}
        if 'scheduling_strategy' not in reduce_ray_remote_args:
            reduce_ray_remote_args = reduce_ray_remote_args.copy()
            reduce_ray_remote_args['scheduling_strategy'] = 'SPREAD'
        shuffle_map = cached_remote_fn(self.map)
        shuffle_reduce = cached_remote_fn(self.reduce)
        should_close_bar = True
        if ctx is not None and ctx.sub_progress_bar_dict is not None:
            bar_name = 'ShuffleMap'
            assert bar_name in ctx.sub_progress_bar_dict, ctx.sub_progress_bar_dict
            map_bar = ctx.sub_progress_bar_dict[bar_name]
            should_close_bar = False
        else:
            map_bar = ProgressBar('Shuffle Map', total=input_num_blocks)
        shuffle_map_out = [shuffle_map.options(**map_ray_remote_args, num_returns=1 + output_num_blocks).remote(i, block, output_num_blocks, *self._map_args) for i, block in enumerate(input_blocks_list)]
        shuffle_map_metadata = []
        for i, refs in enumerate(shuffle_map_out):
            shuffle_map_metadata.append(refs[-1])
            shuffle_map_out[i] = refs[:-1]
        in_blocks_owned_by_consumer = input_blocks._owned_by_consumer
        del input_blocks_list
        if clear_input_blocks:
            input_blocks.clear()
        shuffle_map_metadata = map_bar.fetch_until_complete(shuffle_map_metadata)
        if should_close_bar:
            map_bar.close()
        should_close_bar = True
        if ctx is not None and ctx.sub_progress_bar_dict is not None:
            bar_name = 'ShuffleReduce'
            assert bar_name in ctx.sub_progress_bar_dict, ctx.sub_progress_bar_dict
            reduce_bar = ctx.sub_progress_bar_dict[bar_name]
            should_close_bar = False
        else:
            reduce_bar = ProgressBar('Shuffle Reduce', total=output_num_blocks)
        shuffle_reduce_out = [shuffle_reduce.options(**reduce_ray_remote_args, num_returns=2).remote(*self._reduce_args, *[shuffle_map_out[i][j] for i in range(input_num_blocks)]) for j in range(output_num_blocks)]
        del shuffle_map_out
        new_blocks, new_metadata = zip(*shuffle_reduce_out)
        new_metadata = reduce_bar.fetch_until_complete(list(new_metadata))
        if should_close_bar:
            reduce_bar.close()
        stats = {'map': shuffle_map_metadata, 'reduce': new_metadata}
        return (BlockList(list(new_blocks), list(new_metadata), owned_by_consumer=in_blocks_owned_by_consumer), stats)