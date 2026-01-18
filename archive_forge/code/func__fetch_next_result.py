import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from ray.air._internal.util import StartTraceback
from ray.data import Dataset
from ray.train import Checkpoint, DataConfig
from ray.train._internal.backend_executor import (
from ray.train._internal.session import _TrainingResult, _TrainSession, get_session
from ray.train._internal.utils import ActorWrapper
from ray.train.backend import BackendConfig
from ray.train.base_trainer import (  # noqa: F401
from ray.util.annotations import DeveloperAPI
def _fetch_next_result(self) -> Optional[List[Dict]]:
    """Fetch next results produced by ``session.report()`` from each worker.

        Assumes ``start_training`` has already been called.

        Returns:
            A list of dictionaries of values passed to ``session.report()`` from
                each worker. Each item corresponds to an intermediate result
                a single worker. If there are no more items to fetch,
                returns None.
        """
    results = self._backend_executor.get_next_results()
    if results is None:
        return None
    assert all((isinstance(result, _TrainingResult) for result in results))
    return results