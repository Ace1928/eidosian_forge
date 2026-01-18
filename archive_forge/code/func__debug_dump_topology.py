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
def _debug_dump_topology(topology: Topology, log_to_stdout: bool=True) -> None:
    """Print out current execution state for the topology for debugging.

    Args:
        topology: The topology to debug.
    """
    logger.get_logger(log_to_stdout).info('Execution Progress:')
    for i, (op, state) in enumerate(topology.items()):
        logger.get_logger(log_to_stdout).info(f'{i}: {state.summary_str()}, Blocks Outputted: {state.num_completed_tasks}/{op.num_outputs_total()}')
    logger.get_logger(log_to_stdout).info('')