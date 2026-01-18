import inspect
from typing import Any, Dict, Iterable, Optional, Union
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.one_to_one_operator import AbstractOneToOne
from ray.data.block import UserDefinedFunction
from ray.data.context import DEFAULT_BATCH_SIZE
def _get_udf_name(fn: UserDefinedFunction) -> str:
    try:
        if inspect.isclass(fn):
            return fn.__name__
        elif inspect.ismethod(fn):
            return f'{fn.__self__.__class__.__name__}.{fn.__name__}'
        elif inspect.isfunction(fn):
            return fn.__name__
        else:
            return fn.__class__.__name__
    except AttributeError as e:
        logger.get_logger().error('Failed to get name of UDF %s: %s', fn, e)
        return '<unknown>'