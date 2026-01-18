import copy
from datetime import datetime
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import ray
import ray.cloudpickle as ray_pickle
from ray.air._internal.util import skip_exceptions, exception_cause
from ray.air.constants import (
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.train._internal.storage import StorageContext, _exists_at_fs_path
from ray.train import Checkpoint
from ray.tune.result import (
from ray.tune.utils import UtilMonitor
from ray.tune.utils.log import disable_ipython
from ray.tune.utils.util import Tee
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import DeveloperAPI, PublicAPI
def _close_logfiles(self):
    """Close stdout and stderr logfiles."""
    if self._stderr_logging_handler:
        ray.logger.removeHandler(self._stderr_logging_handler)
    if self._stdout_context:
        self._stdout_stream.flush()
        self._stdout_context.__exit__(None, None, None)
        self._stdout_fp.close()
        self._stdout_context = None
    if self._stderr_context:
        self._stderr_stream.flush()
        self._stderr_context.__exit__(None, None, None)
        self._stderr_fp.close()
        self._stderr_context = None