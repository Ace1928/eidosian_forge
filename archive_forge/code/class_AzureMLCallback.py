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
class AzureMLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [AzureML](https://pypi.org/project/azureml-sdk/).
    """

    def __init__(self, azureml_run=None):
        if not is_azureml_available():
            raise RuntimeError('AzureMLCallback requires azureml to be installed. Run `pip install azureml-sdk`.')
        self.azureml_run = azureml_run

    def on_init_end(self, args, state, control, **kwargs):
        from azureml.core.run import Run
        if self.azureml_run is None and state.is_world_process_zero:
            self.azureml_run = Run.get_context()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.azureml_run and state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.azureml_run.log(k, v, description=k)