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
def _copy_training_args_as_hparams(self, training_args, prefix):
    as_dict = {field.name: getattr(training_args, field.name) for field in fields(training_args) if field.init and (not field.name.endswith('_token'))}
    flat_dict = {str(k): v for k, v in self._clearml.utilities.proxy_object.flatten_dictionary(as_dict).items()}
    self._clearml_task._arguments.copy_from_dict(flat_dict, prefix=prefix)