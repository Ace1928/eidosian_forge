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
def _generate_transform_fn_for_map_rows(fn: UserDefinedFunction) -> MapTransformCallable[Row, Row]:

    def transform_fn(rows: Iterable[Row], _: TaskContext) -> Iterable[Row]:
        for row in rows:
            out_row = fn(row)
            _validate_row_output(out_row)
            yield out_row
    return transform_fn