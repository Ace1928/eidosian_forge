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
def _start_actor(self):
    """Start a new actor and add it to the actor pool as a pending actor."""
    assert self._cls is not None
    ctx = DataContext.get_current()
    actor = self._cls.remote(ctx, src_fn_name=self.name, map_transformer=self._map_transformer)
    res_ref = actor.get_location.remote()

    def _task_done_callback(res_ref):
        has_actor = self._actor_pool.pending_to_running(res_ref)
        if not has_actor:
            return
        self._dispatch_tasks()
    self._submit_metadata_task(res_ref, lambda: _task_done_callback(res_ref))
    self._actor_pool.add_pending_actor(actor, res_ref)