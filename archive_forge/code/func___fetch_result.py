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
def __fetch_result(self, *, remote_actor_ids: List[int], remote_calls: List[ray.ObjectRef], tags: List[str], timeout_seconds: int=None, return_obj_refs: bool=False, mark_healthy: bool=False) -> Tuple[List[ray.ObjectRef], RemoteCallResults]:
    """Try fetching results from remote actor calls.

        Mark whether an actor is healthy or not accordingly.

        Args:
            remote_actor_ids: IDs of the actors these remote
                calls were fired against.
            remote_calls: list of remote calls to fetch.
            tags: list of tags used for identifying the remote calls.
            timeout_seconds: timeout for the ray.wait() call. Default is None.
            return_obj_refs: whether to return ObjectRef instead of actual results.
            mark_healthy: whether to mark certain actors healthy based on the results
                of these remote calls. Useful, for example, to make sure actors
                do not come back without proper state restoration.

        Returns:
            A list of ready ObjectRefs mapping to the results of those calls.
        """
    timeout = float(timeout_seconds) if timeout_seconds is not None else None
    if not remote_calls:
        return ([], RemoteCallResults())
    ready, _ = ray.wait(remote_calls, num_returns=len(remote_calls), timeout=timeout, fetch_local=not return_obj_refs)
    remote_results = RemoteCallResults()
    for r in ready:
        actor_id = remote_actor_ids[remote_calls.index(r)]
        tag = tags[remote_calls.index(r)]
        if return_obj_refs:
            remote_results.add_result(actor_id, ResultOrError(result=r), tag)
            continue
        try:
            result = ray.get(r)
            remote_results.add_result(actor_id, ResultOrError(result=result), tag)
            if mark_healthy and (not self.is_actor_healthy(actor_id)):
                logger.info(f'brining actor {actor_id} back into service.')
                self.set_actor_state(actor_id, healthy=True)
                self._num_actor_restarts += 1
        except Exception as e:
            remote_results.add_result(actor_id, ResultOrError(error=e), tag)
            if isinstance(e, RayError):
                if self.is_actor_healthy(actor_id):
                    logger.error(f'Ray error, taking actor {actor_id} out of service. {str(e)}')
                self.set_actor_state(actor_id, healthy=False)
            else:
                pass
    return (ready, remote_results)