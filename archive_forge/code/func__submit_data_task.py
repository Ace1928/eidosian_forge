import copy
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Union
import ray
from ray import ObjectRef
from ray._raylet import ObjectRefGenerator
from ray.data._internal.compute import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _submit_data_task(self, gen: ObjectRefGenerator, inputs: RefBundle, task_done_callback: Optional[Callable[[], None]]=None):
    """Submit a new data-handling task."""
    task_index = self._next_data_task_idx
    self._next_data_task_idx += 1
    self._metrics.on_task_submitted(task_index, inputs)

    def _output_ready_callback(task_index, output: RefBundle):
        assert len(output) == 1
        self._metrics.on_output_generated(task_index, output)
        self._output_queue.notify_task_output_ready(task_index, output)

    def _task_done_callback(task_index: int, exception: Optional[Exception]):
        self._metrics.on_task_finished(task_index, exception)
        estimated_num_tasks = self.input_dependencies[0].num_outputs_total() / self._metrics.num_inputs_received * self._next_data_task_idx
        self._estimated_output_blocks = round(estimated_num_tasks * self._metrics.num_outputs_of_finished_tasks / self._metrics.num_tasks_finished)
        task = self._data_tasks.pop(task_index)
        self._finished_streaming_gens.append(task.get_waitable())
        self._output_queue.notify_task_completed(task_index)
        if task_done_callback:
            task_done_callback()
    self._data_tasks[task_index] = DataOpTask(task_index, gen, lambda output: _output_ready_callback(task_index, output), functools.partial(_task_done_callback, task_index))