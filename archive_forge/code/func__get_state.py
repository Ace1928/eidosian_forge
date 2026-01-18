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
def _get_state(self) -> ProgressBarState:
    return {'__magic_token__': RAY_TQDM_MAGIC, 'x': self._x, 'pos': self._pos, 'desc': self._desc, 'total': self._total, 'ip': self._ip, 'pid': self._pid, 'uuid': self._uuid, 'closed': self._closed}