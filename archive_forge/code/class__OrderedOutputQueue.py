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
class _OrderedOutputQueue(_OutputQueue):
    """An queue that returns finished tasks in submission order."""

    def __init__(self):
        self._task_outputs: Dict[int, Deque[RefBundle]] = defaultdict(lambda: deque())
        self._current_output_index: int = 0
        self._completed_tasks: Set[int] = set()

    def notify_task_output_ready(self, task_index: int, output: RefBundle):
        self._task_outputs[task_index].append(output)

    def _move_to_next_task(self):
        """Move the outut index to the next task.

        This method should only be called when the current task is complete and all
        outputs have been taken.
        """
        assert len(self._task_outputs[self._current_output_index]) == 0
        assert self._current_output_index in self._completed_tasks
        del self._task_outputs[self._current_output_index]
        self._completed_tasks.remove(self._current_output_index)
        self._current_output_index += 1

    def notify_task_completed(self, task_index: int):
        assert task_index >= self._current_output_index
        self._completed_tasks.add(task_index)
        if task_index == self._current_output_index:
            if len(self._task_outputs[task_index]) == 0:
                self._move_to_next_task()

    def has_next(self) -> bool:
        return len(self._task_outputs[self._current_output_index]) > 0

    def get_next(self) -> RefBundle:
        next_bundle = self._task_outputs[self._current_output_index].popleft()
        if len(self._task_outputs[self._current_output_index]) == 0:
            if self._current_output_index in self._completed_tasks:
                self._move_to_next_task()
        return next_bundle