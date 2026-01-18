from collections import defaultdict
import copy
from dataclasses import dataclass
import logging
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError
from ray.rllib.utils.typing import T
from ray.util.annotations import DeveloperAPI
def _filter_func_and_remote_actor_id_by_state(self, func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]], remote_actor_ids: List[int]):
    """Filter out func and remote worker ids by actor state.

        Args:
            func: A single, or a list of Callables.
            remote_actor_ids: IDs of potential remote workers to apply func on.

        Returns:
            A tuple of (filtered func, filtered remote worker ids).
        """
    if isinstance(func, list):
        assert len(remote_actor_ids) == len(func), 'Func must have the same number of callables as remote actor ids.'
        temp_func = []
        temp_remote_actor_ids = []
        for f, i in zip(func, remote_actor_ids):
            if self.is_actor_healthy(i):
                temp_func.append(f)
                temp_remote_actor_ids.append(i)
        func = temp_func
        remote_actor_ids = temp_remote_actor_ids
    else:
        remote_actor_ids = [i for i in remote_actor_ids if self.is_actor_healthy(i)]
    return (func, remote_actor_ids)