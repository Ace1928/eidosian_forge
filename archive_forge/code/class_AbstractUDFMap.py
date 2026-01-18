import inspect
from typing import Any, Dict, Iterable, Optional, Union
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.one_to_one_operator import AbstractOneToOne
from ray.data.block import UserDefinedFunction
from ray.data.context import DEFAULT_BATCH_SIZE
class AbstractUDFMap(AbstractMap):
    """Abstract class for logical operators performing a UDF that should be converted
    to physical MapOperator.
    """

    def __init__(self, name: str, input_op: LogicalOperator, fn: UserDefinedFunction, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None, min_rows_per_block: Optional[int]=None, compute: Optional[Union[str, ComputeStrategy]]=None, ray_remote_args: Optional[Dict[str, Any]]=None):
        """
        Args:
            name: Name for this operator. This is the name that will appear when
                inspecting the logical plan of a Dataset.
            input_op: The operator preceding this operator in the plan DAG. The outputs
                of `input_op` will be the inputs to this operator.
            fn: User-defined function to be called.
            fn_args: Arguments to `fn`.
            fn_kwargs: Keyword arguments to `fn`.
            fn_constructor_args: Arguments to provide to the initializor of `fn` if
                `fn` is a callable class.
            fn_constructor_kwargs: Keyword Arguments to provide to the initializor of
                `fn` if `fn` is a callable class.
            min_rows_per_block: The target size for blocks outputted by this operator.
            compute: The compute strategy, either ``"tasks"`` (default) to use Ray
                tasks, or ``"actors"`` to use an autoscaling actor pool.
            ray_remote_args: Args to provide to ray.remote.
        """
        name = f'{name}({_get_udf_name(fn)})'
        super().__init__(name, input_op, ray_remote_args)
        self._fn = fn
        self._fn_args = fn_args
        self._fn_kwargs = fn_kwargs
        self._fn_constructor_args = fn_constructor_args
        self._fn_constructor_kwargs = fn_constructor_kwargs
        self._min_rows_per_block = min_rows_per_block
        self._compute = compute or TaskPoolStrategy()