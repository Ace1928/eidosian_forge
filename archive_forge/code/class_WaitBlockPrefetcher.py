import logging
import threading
from contextlib import nullcontext
from typing import Any, Callable, Iterator, List, Optional, Tuple
import ray
from ray.actor import ActorHandle
from ray.data._internal.batcher import Batcher, ShufflingBatcher
from ray.data._internal.block_batching.interfaces import (
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class WaitBlockPrefetcher(BlockPrefetcher):
    """Block prefetcher using ray.wait."""

    def __init__(self):
        self._blocks = []
        self._stopped = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._run, name='Prefetcher', daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            try:
                blocks_to_wait = []
                with self._condition:
                    if len(self._blocks) > 0:
                        blocks_to_wait, self._blocks = (self._blocks[:], [])
                    else:
                        if self._stopped:
                            return
                        blocks_to_wait = []
                        self._condition.wait()
                if len(blocks_to_wait) > 0:
                    ray.wait(blocks_to_wait, num_returns=1, fetch_local=True)
            except Exception:
                logger.exception('Error in prefetcher thread.')

    def prefetch_blocks(self, blocks: List[ObjectRef[Block]]):
        with self._condition:
            if self._stopped:
                raise RuntimeError('Prefetcher is stopped.')
            self._blocks = blocks
            self._condition.notify()

    def stop(self):
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            self._condition.notify()

    def __del__(self):
        self.stop()