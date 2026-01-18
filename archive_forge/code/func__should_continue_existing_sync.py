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
def _should_continue_existing_sync(self):
    """Returns whether a previous sync is still running within the timeout."""
    return self._sync_process and self._sync_process.is_running and (time.time() - self._sync_process.start_time < self.sync_timeout)