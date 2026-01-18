import collections
from types import GeneratorType
from typing import Any, Callable, Iterable, Iterator, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from ray.data._internal.compute import get_compute
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.execution.util import make_callable_class_concurrent
from ray.data._internal.logical.operators.map_operator import (
from ray.data._internal.numpy_support import is_valid_udf_return
from ray.data._internal.util import _truncated_repr
from ray.data.block import (
from ray.data.context import DataContext
def _generate_transform_fn_for_map_batches(fn: UserDefinedFunction) -> MapTransformCallable[DataBatch, DataBatch]:

    def transform_fn(batches: Iterable[DataBatch], _: TaskContext) -> Iterable[DataBatch]:
        for batch in batches:
            try:
                if not isinstance(batch, collections.abc.Mapping) and BlockAccessor.for_block(batch).num_rows() == 0:
                    res = [batch]
                else:
                    res = fn(batch)
                    if not isinstance(res, GeneratorType):
                        res = [res]
            except ValueError as e:
                read_only_msgs = ['assignment destination is read-only', 'buffer source array is read-only']
                err_msg = str(e)
                if any((msg in err_msg for msg in read_only_msgs)):
                    raise ValueError(f"Batch mapper function {fn.__name__} tried to mutate a zero-copy read-only batch. To be able to mutate the batch, pass zero_copy_batch=False to map_batches(); this will create a writable copy of the batch before giving it to fn. To elide this copy, modify your mapper function so it doesn't try to mutate its input.") from e
                else:
                    raise e from None
            else:
                for out_batch in res:
                    _validate_batch_output(out_batch)
                    yield out_batch
    return transform_fn