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
def _get_or_refresh_resource_limits(self) -> ExecutionResources:
    """Return concrete limits for use at the current time.

        This method autodetects any unspecified execution resource limits based on the
        current cluster size, refreshing these values periodically to support cluster
        autoscaling.
        """
    base = self._options.resource_limits
    exclude = self._options.exclude_resources
    cluster = ray.cluster_resources()
    cpu = base.cpu
    if cpu is None:
        cpu = cluster.get('CPU', 0.0) - (exclude.cpu or 0.0)
    gpu = base.gpu
    if gpu is None:
        gpu = cluster.get('GPU', 0.0) - (exclude.gpu or 0.0)
    object_store_memory = base.object_store_memory
    if object_store_memory is None:
        object_store_memory = round(DEFAULT_OBJECT_STORE_MEMORY_LIMIT_FRACTION * cluster.get('object_store_memory', 0.0)) - (exclude.object_store_memory or 0)
    return ExecutionResources(cpu=cpu, gpu=gpu, object_store_memory=object_store_memory)