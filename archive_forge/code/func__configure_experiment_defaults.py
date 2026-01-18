import os
from typing import Dict, List
import pyarrow.fs
from ray.tune.logger import LoggerCallback
from ray.tune.experiment import Trial
from ray.tune.utils import flatten_dict
def _configure_experiment_defaults(self):
    """Disable the specific autologging features that cause throttling."""
    for option in self._exclude_autolog:
        if not self.experiment_kwargs.get(option):
            self.experiment_kwargs[option] = False