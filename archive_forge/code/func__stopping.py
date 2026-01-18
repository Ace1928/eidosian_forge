import json
import os
import random
import string
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk import wandb_sweep
from wandb.sdk.launch.sweeps.utils import (
from wandb.util import get_module
def _stopping(self) -> List[sweeps.SweepRun]:
    if 'early_terminate' not in self.sweep_config:
        return []
    stopper = self._custom_stopping or sweeps.stop_runs
    stop_runs = stopper(self._sweep_config, self._sweep_runs or [])
    debug_lines = '\n'.join([' '.join([f'{k}={v}' for k, v in run.early_terminate_info.items()]) for run in stop_runs if run.early_terminate_info is not None])
    if debug_lines:
        self._log_debug += debug_lines
    return stop_runs