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
@DeveloperAPI
def add_actors(self, actors: List[ActorHandle]):
    """Add a list of actors to the pool.

        Args:
            actors: A list of ray remote actors to be added to the pool.
        """
    for actor in actors:
        self.__actors[self.__next_id] = actor
        self.__remote_actor_states[self.__next_id] = self._ActorState()
        self.__next_id += 1