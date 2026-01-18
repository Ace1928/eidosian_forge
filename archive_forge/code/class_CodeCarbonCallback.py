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
class CodeCarbonCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that tracks the CO2 emission of training.
    """

    def __init__(self):
        if not is_codecarbon_available():
            raise RuntimeError('CodeCarbonCallback requires `codecarbon` to be installed. Run `pip install codecarbon`.')
        import codecarbon
        self._codecarbon = codecarbon
        self.tracker = None

    def on_init_end(self, args, state, control, **kwargs):
        if self.tracker is None and state.is_local_process_zero:
            self.tracker = self._codecarbon.EmissionsTracker(output_dir=args.output_dir)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.tracker and state.is_local_process_zero:
            self.tracker.start()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tracker and state.is_local_process_zero:
            self.tracker.stop()