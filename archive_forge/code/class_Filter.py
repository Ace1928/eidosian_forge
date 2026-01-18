import inspect
from typing import Any, Dict, Iterable, Optional, Union
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.one_to_one_operator import AbstractOneToOne
from ray.data.block import UserDefinedFunction
from ray.data.context import DEFAULT_BATCH_SIZE
class Filter(AbstractUDFMap):
    """Logical operator for filter."""

    def __init__(self, input_op: LogicalOperator, fn: UserDefinedFunction, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        super().__init__('Filter', input_op, fn, compute=compute, ray_remote_args=ray_remote_args)

    @property
    def can_modify_num_rows(self) -> bool:
        return True