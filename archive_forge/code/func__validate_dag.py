import os
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.streaming_executor_state import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.context import DataContext
def _validate_dag(dag: PhysicalOperator, limits: ExecutionResources) -> None:
    """Raises an exception on invalid DAGs.

    It checks if the the sum of min actor pool sizes are larger than the resource
    limit, as well as other unsupported resource configurations.

    This should be called prior to creating the topology from the DAG.

    Args:
        dag: The DAG to validate.
        limits: The limits to validate against.
    """
    seen = set()

    def walk(op):
        seen.add(op)
        for parent in op.input_dependencies:
            if parent not in seen:
                yield from walk(parent)
        yield op
    base_usage = ExecutionResources(cpu=1)
    for op in walk(dag):
        base_usage = base_usage.add(op.base_resource_usage())
    if not base_usage.satisfies_limit(limits):
        error_message = "The current cluster doesn't have the required resources to execute your Dataset pipeline:\n"
        if base_usage.cpu is not None and limits.cpu is not None and (base_usage.cpu > limits.cpu):
            error_message += f'- Your application needs {base_usage.cpu} CPU(s), but your cluster only has {limits.cpu}.\n'
        if base_usage.gpu is not None and limits.gpu is not None and (base_usage.gpu > limits.gpu):
            error_message += f'- Your application needs {base_usage.gpu} GPU(s), but your cluster only has {limits.gpu}.\n'
        if base_usage.object_store_memory is not None and base_usage.object_store_memory is not None and (base_usage.object_store_memory > limits.object_store_memory):
            error_message += f'- Your application needs {base_usage.object_store_memory}B object store memory, but your cluster only has {limits.object_store_memory}B.\n'
        raise ValueError(error_message.strip())