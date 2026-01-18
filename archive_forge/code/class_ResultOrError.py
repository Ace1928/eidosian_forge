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
class ResultOrError:
    """A wrapper around a result or an error.

    This is used to return data from FaultTolerantActorManager
    that allows us to distinguish between error and actual results.
    """

    def __init__(self, result: Any=None, error: Exception=None):
        """One and only one of result or error should be set.

        Args:
            result: The result of the computation.
            error: Alternatively, the error that occurred during the computation.
        """
        self._result = result
        self._error = error.as_instanceof_cause() if isinstance(error, RayTaskError) else error

    @property
    def ok(self):
        return self._error is None

    def get(self):
        """Returns the result or the error."""
        if self._error:
            return self._error
        else:
            return self._result