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
def _validate_batch_output(batch: Block) -> None:
    if not isinstance(batch, (list, pa.Table, np.ndarray, collections.abc.Mapping, pd.core.frame.DataFrame)):
        raise ValueError(f"The `fn` you passed to `map_batches` returned a value of type {type(batch)}. This isn't allowed -- `map_batches` expects `fn` to return a `pandas.DataFrame`, `pyarrow.Table`, `numpy.ndarray`, `list`, or `dict[str, numpy.ndarray]`.")
    if isinstance(batch, list):
        raise ValueError(f"Error validating {_truncated_repr(batch)}: Returning a list of objects from `map_batches` is not allowed in Ray 2.5. To return Python objects, wrap them in a named dict field, e.g., return `{{'results': objects}}` instead of just `objects`.")
    if isinstance(batch, collections.abc.Mapping):
        for key, value in list(batch.items()):
            if not is_valid_udf_return(value):
                raise ValueError(f'Error validating {_truncated_repr(batch)}: The `fn` you passed to `map_batches` returned a `dict`. `map_batches` expects all `dict` values to be `list` or `np.ndarray` type, but the value corresponding to key {key!r} is of type {type(value)}. To fix this issue, convert the {type(value)} to a `np.ndarray`.')