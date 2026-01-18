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
class RemoteCallResults:
    """Represents a list of results from calls to a set of actors.

    CallResults provides convenient APIs to iterate over the results
    while skipping errors, etc.

    .. testcode::
        :skipif: True

        manager = FaultTolerantActorManager(
            actors, max_remote_requests_in_flight_per_actor=2,
        )
        results = manager.foreach_actor(lambda w: w.call())

        # Iterate over all results ignoring errors.
        for result in results.ignore_errors():
            print(result.get())
    """

    class _Iterator:
        """An iterator over the results of a remote call."""

        def __init__(self, call_results: List[CallResult]):
            self._call_results = call_results

        def __iter__(self) -> Iterator[CallResult]:
            return self

        def __next__(self) -> CallResult:
            if not self._call_results:
                raise StopIteration
            return self._call_results.pop(0)

    def __init__(self):
        self.result_or_errors: List[CallResult] = []

    def add_result(self, actor_id: int, result_or_error: ResultOrError, tag: str):
        """Add index of a remote actor plus the call result to the list.

        Args:
            actor_id: ID of the remote actor.
            result_or_error: The result or error from the call.
            tag: A description to identify the call.
        """
        self.result_or_errors.append(CallResult(actor_id, result_or_error, tag))

    def __iter__(self) -> Iterator[ResultOrError]:
        """Return an iterator over the results."""
        return self._Iterator(copy.copy(self.result_or_errors))

    def ignore_errors(self) -> Iterator[ResultOrError]:
        """Return an iterator over the results, skipping all errors."""
        return self._Iterator([r for r in self.result_or_errors if r.ok])

    def ignore_ray_errors(self) -> Iterator[ResultOrError]:
        """Return an iterator over the results, skipping only Ray errors.

        Similar to ignore_errors, but only skips Errors raised because of
        remote actor problems (often get restored automatcially).
        This is useful for callers that wants to handle application errors differently.
        """
        return self._Iterator([r for r in self.result_or_errors if not isinstance(r.get(), RayActorError)])