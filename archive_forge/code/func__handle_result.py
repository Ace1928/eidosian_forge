import enum
import os
import pickle
import urllib
import warnings
import numpy as np
from numbers import Number
import pyarrow.fs
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ray
from ray import logger
from ray.air import session
from ray.air._internal import usage as air_usage
from ray.air.util.node import _force_on_current_node
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray._private.storage import _load_class
from ray.util import PublicAPI
from ray.util.queue import Queue
def _handle_result(self, result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.get('config', {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter='/')
    for k, v in flat_result.items():
        if any((k.startswith(item + '/') or k == item for item in self._exclude)):
            continue
        elif any((k.startswith(item + '/') or k == item for item in self._to_config)):
            config_update[k] = v
        elif not _is_allowed_type(v):
            continue
        else:
            log[k] = v
    config_update.pop('callbacks', None)
    return (log, config_update)