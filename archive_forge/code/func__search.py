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
def _search(self) -> Optional[sweeps.SweepRun]:
    search = self._custom_search or sweeps.next_run
    next_run = search(self._sweep_config, self._sweep_runs or [])
    if next_run is None:
        self._done_scheduling = True
    return next_run