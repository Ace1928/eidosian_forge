import copy
import logging
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.execution.interfaces import NodeIdStr, RefBundle
from ray.data._internal.execution.legacy_compat import execute_to_legacy_bundle_iterator
from ray.data._internal.execution.operators.output_splitter import OutputSplitter
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import create_dataset_tag
from ray.data.block import Block, BlockMetadata
from ray.data.iterator import DataIterator
from ray.types import ObjectRef
from ray.util.debug import log_once
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _barrier(self, split_idx: int) -> int:
    """Arrive and block until the start of the given epoch."""
    with self._lock:
        starting_epoch = self._cur_epoch
        self._unfinished_clients_in_epoch -= 1
    start_time = time.time()
    while self._cur_epoch == starting_epoch and self._unfinished_clients_in_epoch != 0:
        if time.time() - start_time > BLOCKED_CLIENT_WARN_TIMEOUT:
            if log_once(f'stream_split_blocked_{split_idx}_{starting_epoch}'):
                logger.warning(f'StreamSplitDataIterator(epoch={starting_epoch}, split={split_idx}) blocked waiting on other clients for more than {BLOCKED_CLIENT_WARN_TIMEOUT}s. All clients must read from the DataIterator splits at the same time. This warning will not be printed again for this epoch.')
        time.sleep(0.1)
    with self._lock:
        if self._cur_epoch == starting_epoch:
            self._cur_epoch += 1
            self._unfinished_clients_in_epoch = self._n
            self._output_iterator = next(self._next_epoch)
    assert self._output_iterator is not None
    return starting_epoch + 1