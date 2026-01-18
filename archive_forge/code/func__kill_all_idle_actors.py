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
def _kill_all_idle_actors(self):
    idle_actors = [actor for actor, tasks_in_flight in self._num_tasks_in_flight.items() if tasks_in_flight == 0]
    for actor in idle_actors:
        self._kill_running_actor(actor)
    self._should_kill_idle_actors = True