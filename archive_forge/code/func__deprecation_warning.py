import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
def _deprecation_warning(self, attr_name: str, extra_msg: str):
    if getattr(self, attr_name) != _DEPRECATED_VALUE:
        if log_once(f'sync_config_param_deprecation_{attr_name}'):
            warnings.warn(f'`SyncConfig({attr_name})` is a deprecated configuration and will be ignored. Please remove it from your `SyncConfig`, as this will raise an error in a future version of Ray.{extra_msg}')