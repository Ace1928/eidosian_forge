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
def _init_summary_writer(self, args, log_dir=None):
    log_dir = log_dir or args.logging_dir
    if self._SummaryWriter is not None:
        self.tb_writer = self._SummaryWriter(log_dir=log_dir)