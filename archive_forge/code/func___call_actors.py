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
def __call_actors(self, func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]], *, remote_actor_ids: List[int]=None) -> List[ray.ObjectRef]:
    """Apply functions on a list of remote actors.

        Args:
            func: A single, or a list of Callables, that get applied on the list
                of specified remote actors.
            remote_actor_ids: Apply func on this selected set of remote actors.

        Returns:
            A list of ObjectRefs returned from the remote calls.
        """
    if isinstance(func, list):
        assert len(remote_actor_ids) == len(func), 'Funcs must have the same number of callables as actor indices.'
    if remote_actor_ids is None:
        remote_actor_ids = self.actor_ids()
    if isinstance(func, list):
        calls = [self.__actors[i].apply.remote(f) for i, f in zip(remote_actor_ids, func)]
    else:
        calls = [self.__actors[i].apply.remote(func) for i in remote_actor_ids]
    return calls