import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
def _ensure_run_with_monitoring(self):
    if self._initial_run is not None:
        self._use_initial_run()
    else:
        if not self._force_reset_monitoring_run and self._is_monitoring_run:
            return
        if self._run and (not self._is_monitoring_run) and (not self._force_reset_monitoring_run):
            self._initialize_run(with_id=self._run_id)
            self._is_monitoring_run = True
        else:
            self._initialize_run()
            self._force_reset_monitoring_run = False