import inspect
from typing import Any, Dict, Iterable, Optional, Union
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.one_to_one_operator import AbstractOneToOne
from ray.data.block import UserDefinedFunction
from ray.data.context import DEFAULT_BATCH_SIZE
class AbstractMap(AbstractOneToOne):
    """Abstract class for logical operators that should be converted to physical
    MapOperator.
    """

    def __init__(self, name: str, input_op: Optional[LogicalOperator]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        """
        Args:
            name: Name for this operator. This is the name that will appear when
                inspecting the logical plan of a Dataset.
            input_op: The operator preceding this operator in the plan DAG. The outputs
                of `input_op` will be the inputs to this operator.
            ray_remote_args: Args to provide to ray.remote.
        """
        super().__init__(name, input_op)
        self._ray_remote_args = ray_remote_args or {}