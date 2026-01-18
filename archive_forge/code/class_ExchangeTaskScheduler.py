from typing import Any, Dict, List, Optional, Tuple, Union
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockMetadata
class ExchangeTaskScheduler:
    """
    An interface to schedule exchange tasks (`exchange_spec`) for multi-nodes
    execution.
    """

    def __init__(self, exchange_spec: ExchangeTaskSpec):
        """
        Args:
            exchange_spec: The implementation of exchange tasks to execute.
        """
        self._exchange_spec = exchange_spec

    def execute(self, refs: List[RefBundle], output_num_blocks: int, map_ray_remote_args: Optional[Dict[str, Any]]=None, reduce_ray_remote_args: Optional[Dict[str, Any]]=None) -> Tuple[List[RefBundle], StatsDict]:
        """
        Execute the exchange tasks on input `refs`.
        """
        raise NotImplementedError