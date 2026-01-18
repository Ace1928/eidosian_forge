import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
from ray.data.context import DataContext
class OpBufferQueue:
    """A FIFO queue to buffer RefBundles between upstream and downstream operators.
    This class is thread-safe.
    """

    def __init__(self):
        self._memory_usage = 0
        self._num_blocks = 0
        self._queue = deque()
        self._num_per_split = defaultdict(int)
        self._lock = threading.Lock()
        super().__init__()

    @property
    def memory_usage(self) -> int:
        """The total memory usage of the queue in bytes."""
        with self._lock:
            return self._memory_usage

    @property
    def num_blocks(self) -> int:
        """The total number of blocks in the queue."""
        with self._lock:
            return self._num_blocks

    def __len__(self):
        return len(self._queue)

    def has_next(self, output_split_idx: Optional[int]=None) -> bool:
        """Whether next RefBundle is available.

        Args:
            output_split_idx: If specified, only check ref bundles with the
                given output split.
        """
        if output_split_idx is None:
            return len(self._queue) > 0
        else:
            with self._lock:
                return self._num_per_split[output_split_idx] > 0

    def append(self, ref: RefBundle):
        """Append a RefBundle to the queue."""
        self._queue.append(ref)
        with self._lock:
            self._memory_usage += ref.size_bytes()
            self._num_blocks += len(ref.blocks)
            if ref.output_split_idx is not None:
                self._num_per_split[ref.output_split_idx] += 1

    def pop(self, output_split_idx: Optional[int]=None) -> Optional[RefBundle]:
        """Pop a RefBundle from the queue.
        Args:
            output_split_idx: If specified, only pop a RefBundle
                with the given output split.
        Returns:
            A RefBundle if available, otherwise None.
        """
        ret = None
        if output_split_idx is None:
            try:
                ret = self._queue.popleft()
            except IndexError:
                pass
        else:
            for i in range(len(self._queue)):
                ref = self._queue[i]
                if ref.output_split_idx == output_split_idx:
                    ret = ref
                    del self._queue[i]
                    break
        if ret is None:
            return None
        with self._lock:
            self._memory_usage -= ret.size_bytes()
            self._num_blocks -= len(ret.blocks)
            if ret.output_split_idx is not None:
                self._num_per_split[ret.output_split_idx] -= 1
        return ret

    def clear(self):
        with self._lock:
            self._queue.clear()
            self._memory_usage = 0
            self._num_blocks = 0
            self._num_per_split.clear()