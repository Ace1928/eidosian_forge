import logging
import os
from packaging import version
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional
from ray._private.dict import flatten_dict
def _start_active_run(self, run_name: Optional[str]=None, tags: Optional[Dict]=None) -> 'Run':
    """Starts a run and sets it as the active run if one does not exist.

        If an active run already exists, then returns it.
        """
    active_run = self._mlflow.active_run()
    if active_run:
        return active_run
    return self._mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id, tags=tags)