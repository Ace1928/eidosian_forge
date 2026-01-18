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
class CometCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Comet ML](https://www.comet.ml/site/).
    """

    def __init__(self):
        if not _has_comet:
            raise RuntimeError('CometCallback requires comet-ml to be installed. Run `pip install comet-ml`.')
        self._initialized = False
        self._log_assets = False

    def setup(self, args, state, model):
        """
        Setup the optional Comet.ml integration.

        Environment:
        - **COMET_MODE** (`str`, *optional*, defaults to `ONLINE`):
            Whether to create an online, offline experiment or disable Comet logging. Can be `OFFLINE`, `ONLINE`, or
            `DISABLED`.
        - **COMET_PROJECT_NAME** (`str`, *optional*):
            Comet project name for experiments.
        - **COMET_OFFLINE_DIRECTORY** (`str`, *optional*):
            Folder to use for saving offline experiments when `COMET_MODE` is `OFFLINE`.
        - **COMET_LOG_ASSETS** (`str`, *optional*, defaults to `TRUE`):
            Whether or not to log training assets (tf event logs, checkpoints, etc), to Comet. Can be `TRUE`, or
            `FALSE`.

        For a number of configurable items in the environment, see
        [here](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).
        """
        self._initialized = True
        log_assets = os.getenv('COMET_LOG_ASSETS', 'FALSE').upper()
        if log_assets in {'TRUE', '1'}:
            self._log_assets = True
        if state.is_world_process_zero:
            comet_mode = os.getenv('COMET_MODE', 'ONLINE').upper()
            experiment = None
            experiment_kwargs = {'project_name': os.getenv('COMET_PROJECT_NAME', 'huggingface')}
            if comet_mode == 'ONLINE':
                experiment = comet_ml.Experiment(**experiment_kwargs)
                experiment.log_other('Created from', 'transformers')
                logger.info('Automatic Comet.ml online logging enabled')
            elif comet_mode == 'OFFLINE':
                experiment_kwargs['offline_directory'] = os.getenv('COMET_OFFLINE_DIRECTORY', './')
                experiment = comet_ml.OfflineExperiment(**experiment_kwargs)
                experiment.log_other('Created from', 'transformers')
                logger.info('Automatic Comet.ml offline logging enabled; use `comet upload` when finished')
            if experiment is not None:
                experiment._set_model_graph(model, framework='transformers')
                experiment._log_parameters(args, prefix='args/', framework='transformers')
                if hasattr(model, 'config'):
                    experiment._log_parameters(model.config, prefix='config/', framework='transformers')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            experiment = comet_ml.config.get_global_experiment()
            if experiment is not None:
                experiment._log_metrics(logs, step=state.global_step, epoch=state.epoch, framework='transformers')

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            experiment = comet_ml.config.get_global_experiment()
            if experiment is not None:
                if self._log_assets is True:
                    logger.info('Logging checkpoints. This may take time.')
                    experiment.log_asset_folder(args.output_dir, recursive=True, log_file_name=True, step=state.global_step)
                experiment.end()