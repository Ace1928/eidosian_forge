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
class DVCLiveCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [DVCLive](https://www.dvc.org/doc/dvclive).

    Use the environment variables below in `setup` to configure the integration. To customize this callback beyond
    those environment variables, see [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

    Args:
        live (`dvclive.Live`, *optional*, defaults to `None`):
            Optional Live instance. If None, a new instance will be created using **kwargs.
        log_model (Union[Literal["all"], bool], *optional*, defaults to `None`):
            Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [`Trainer`]. If set to `True`,
            the final checkpoint is logged at the end of training. If set to `"all"`, the entire
            [`TrainingArguments`]'s `output_dir` is logged at each checkpoint.
    """

    def __init__(self, live: Optional[Any]=None, log_model: Optional[Union[Literal['all'], bool]]=None, **kwargs):
        if not is_dvclive_available():
            raise RuntimeError('DVCLiveCallback requires dvclive to be installed. Run `pip install dvclive`.')
        from dvclive import Live
        self._initialized = False
        self.live = None
        if isinstance(live, Live):
            self.live = live
        elif live is not None:
            raise RuntimeError(f'Found class {live.__class__} for live, expected dvclive.Live')
        self._log_model = log_model
        if self._log_model is None:
            log_model_env = os.getenv('HF_DVCLIVE_LOG_MODEL', 'FALSE')
            if log_model_env.upper() in ENV_VARS_TRUE_VALUES:
                self._log_model = True
            elif log_model_env.lower() == 'all':
                self._log_model = 'all'

    def setup(self, args, state, model):
        """
        Setup the optional DVCLive integration. To customize this callback beyond the environment variables below, see
        [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

        Environment:
        - **HF_DVCLIVE_LOG_MODEL** (`str`, *optional*):
            Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [`Trainer`]. If set to `True` or
            *1*, the final checkpoint is logged at the end of training. If set to `all`, the entire
            [`TrainingArguments`]'s `output_dir` is logged at each checkpoint.
        """
        from dvclive import Live
        self._initialized = True
        if state.is_world_process_zero:
            if not self.live:
                self.live = Live()
            self.live.log_params(args.to_dict())

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            from dvclive.plots import Metric
            from dvclive.utils import standardize_metric_name
            for key, value in logs.items():
                if Metric.could_log(value):
                    self.live.log_metric(standardize_metric_name(key, 'dvclive.huggingface'), value)
                else:
                    logger.warning(f'''Trainer is attempting to log a value of "{value}" of type {type(value)} for key "{key}" as a scalar. This invocation of DVCLive's Live.log_metric() is incorrect so we dropped this attribute.''')
            self.live.next_step()

    def on_save(self, args, state, control, **kwargs):
        if self._log_model == 'all' and self._initialized and state.is_world_process_zero:
            self.live.log_artifact(args.output_dir)

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            from transformers.trainer import Trainer
            if self._log_model is True:
                fake_trainer = Trainer(args=args, model=kwargs.get('model'), tokenizer=kwargs.get('tokenizer'))
                name = 'best' if args.load_best_model_at_end else 'last'
                output_dir = os.path.join(args.output_dir, name)
                fake_trainer.save_model(output_dir)
                self.live.log_artifact(output_dir, name=name, type='model', copy=True)
            self.live.end()