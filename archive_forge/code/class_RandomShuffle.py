from typing import Any, Dict, List, Optional
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.planner.exchange.sort_task_spec import SortTaskSpec
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn
class RandomShuffle(AbstractAllToAll):
    """Logical operator for random_shuffle."""

    def __init__(self, input_op: LogicalOperator, name: str='RandomShuffle', seed: Optional[int]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        super().__init__(name, input_op, sub_progress_bar_names=[ExchangeTaskSpec.MAP_SUB_PROGRESS_BAR_NAME, ExchangeTaskSpec.REDUCE_SUB_PROGRESS_BAR_NAME], ray_remote_args=ray_remote_args)
        self._seed = seed