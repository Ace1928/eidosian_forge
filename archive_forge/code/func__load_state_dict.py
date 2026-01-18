from typing import Any, Dict, Optional
import warnings
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from .storage import (
from .planner import LoadPlanner
from .default_planner import DefaultLoadPlanner
from .utils import _DistWrapper, _all_gather_keys
def _load_state_dict(state_dict: Dict[str, Any], storage_reader: StorageReader, process_group: Optional[dist.ProcessGroup]=None, coordinator_rank: int=0, no_dist: bool=False, planner: Optional[LoadPlanner]=None) -> None:
    torch._C._log_api_usage_once('torch.distributed.checkpoint.load_state_dict')
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    def local_step():
        assert planner is not None
        metadata = storage_reader.read_metadata()
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans
    central_plan = distW.reduce_scatter('plan', local_step, global_step)

    def read_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)
        all_reads.wait()
        return None
    _ = distW.all_gather('read', read_data)