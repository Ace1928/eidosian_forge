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
def _start_if_not_started(self) -> None:
    if self._started:
        return
    if self._defer_sweep_creation:
        raise ControllerError('Must specify or create a sweep before running controller.')
    obj = self._sweep_object_read_from_backend()
    if not obj:
        return
    is_local = self._sweep_config.get('controller', {}).get('type') == 'local'
    if not is_local:
        raise ControllerError('Only sweeps with a local controller are currently supported.')
    self._started = True
    self._controller = {}
    self._sweep_object_sync_to_backend()