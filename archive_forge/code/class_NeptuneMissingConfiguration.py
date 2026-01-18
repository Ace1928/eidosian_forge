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
class NeptuneMissingConfiguration(Exception):

    def __init__(self):
        super().__init__('\n        ------ Unsupported ---- We were not able to create new runs. You provided a custom Neptune run to\n        `NeptuneCallback` with the `run` argument. For the integration to work fully, provide your `api_token` and\n        `project` by saving them as environment variables or passing them to the callback.\n        ')