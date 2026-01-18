import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
def _dump_state(self) -> None:
    if ray._private.worker.global_worker.mode == ray.WORKER_MODE:
        print(json.dumps(self._get_state()) + '\n', end='')
    else:
        instance().process_state_update(copy.deepcopy(self._get_state()))