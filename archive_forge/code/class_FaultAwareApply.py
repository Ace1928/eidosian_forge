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
class FaultAwareApply:

    @DeveloperAPI
    def ping(self) -> str:
        """Ping the actor. Can be used as a health check.

        Returns:
            "pong" if actor is up and well.
        """
        return 'pong'

    @DeveloperAPI
    def apply(self, func: Callable[[Any, Optional[Any], Optional[Any]], T], *args, **kwargs) -> T:
        """Calls the given function with this Actor instance.

        A generic interface for applying arbitray member functions on a
        remote actor.

        Args:
            func: The function to call, with this RolloutWorker as first
                argument, followed by args, and kwargs.
            args: Optional additional args to pass to the function call.
            kwargs: Optional additional kwargs to pass to the function call.

        Returns:
            The return value of the function call.
        """
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if self.config.recreate_failed_workers:
                logger.exception('Worker exception, recreating: {}'.format(e))
                time.sleep(self.config.delay_between_worker_restarts_s)
                sys.exit(1)
            else:
                raise e