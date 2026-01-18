import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def _dispatch_tasks(self):
    """Try to dispatch tasks from the bundle buffer to the actor pool.

        This is called when:
            * a new input bundle is added,
            * a task finishes,
            * a new worker has been created.
        """
    while self._bundle_queue:
        if self._actor_locality_enabled:
            actor = self._actor_pool.pick_actor(self._bundle_queue[0])
        else:
            actor = self._actor_pool.pick_actor()
        if actor is None:
            break
        bundle = self._bundle_queue.popleft()
        input_blocks = [block for block, _ in bundle.blocks]
        ctx = TaskContext(task_idx=self._next_data_task_idx, target_max_block_size=self.actual_target_max_block_size)
        gen = actor.submit.options(num_returns='streaming', name=self.name).remote(DataContext.get_current(), ctx, *input_blocks)

        def _task_done_callback(actor_to_return):
            self._actor_pool.return_actor(actor_to_return)
            self._dispatch_tasks()
        actor_to_return = actor
        self._submit_data_task(gen, bundle, lambda: _task_done_callback(actor_to_return))
    if self._bundle_queue:
        self._scale_up_if_needed()
    else:
        self._scale_down_if_needed()