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
def _create_map_transformer_for_row_based_map_op(row_fn: MapTransformCallable[Row, Row], init_fn: Optional[Callable[[], None]]=None) -> MapTransformer:
    """Create a MapTransformer for a row-based map operator
    (e.g. map, flat_map, filter)."""
    transform_fns = [BlocksToRowsMapTransformFn.instance(), RowMapTransformFn(row_fn), BuildOutputBlocksMapTransformFn.for_rows()]
    return MapTransformer(transform_fns, init_fn=init_fn)