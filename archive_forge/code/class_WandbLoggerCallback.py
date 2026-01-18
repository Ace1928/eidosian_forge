import enum
import os
import pickle
import urllib
import warnings
import numpy as np
from numbers import Number
import pyarrow.fs
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ray
from ray import logger
from ray.air import session
from ray.air._internal import usage as air_usage
from ray.air.util.node import _force_on_current_node
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray._private.storage import _load_class
from ray.util import PublicAPI
from ray.util.queue import Queue
class WandbLoggerCallback(LoggerCallback):
    """WandbLoggerCallback

    Weights and biases (https://www.wandb.ai/) is a tool for experiment
    tracking, model optimization, and dataset versioning. This Ray Tune
    ``LoggerCallback`` sends metrics to Wandb for automatic tracking and
    visualization.

    Example:

        .. testcode::

            import random

            from ray import train, tune
            from ray.train import RunConfig
            from ray.air.integrations.wandb import WandbLoggerCallback


            def train_func(config):
                offset = random.random() / 5
                for epoch in range(2, config["epochs"]):
                    acc = 1 - (2 + config["lr"]) ** -epoch - random.random() / epoch - offset
                    loss = (2 + config["lr"]) ** -epoch + random.random() / epoch + offset
                    train.report({"acc": acc, "loss": loss})


            tuner = tune.Tuner(
                train_func,
                param_space={
                    "lr": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
                    "epochs": 10,
                },
                run_config=RunConfig(
                    callbacks=[WandbLoggerCallback(project="Optimization_Project")]
                ),
            )
            results = tuner.fit()

        .. testoutput::
            :hide:

            ...

    Args:
        project: Name of the Wandb project. Mandatory.
        group: Name of the Wandb group. Defaults to the trainable
            name.
        api_key_file: Path to file containing the Wandb API KEY. This
            file only needs to be present on the node running the Tune script
            if using the WandbLogger.
        api_key: Wandb API Key. Alternative to setting ``api_key_file``.
        excludes: List of metrics and config that should be excluded from
            the log.
        log_config: Boolean indicating if the ``config`` parameter of
            the ``results`` dict should be logged. This makes sense if
            parameters will change during training, e.g. with
            PopulationBasedTraining. Defaults to False.
        upload_checkpoints: If ``True``, model checkpoints will be uploaded to
            Wandb as artifacts. Defaults to ``False``.
        **kwargs: The keyword arguments will be pased to ``wandb.init()``.

    Wandb's ``group``, ``run_id`` and ``run_name`` are automatically selected
    by Tune, but can be overwritten by filling out the respective configuration
    values.

    Please see here for all other valid configuration settings:
    https://docs.wandb.ai/library/init
    """
    _exclude_results = ['done', 'should_checkpoint']
    AUTO_CONFIG_KEYS = ['trial_id', 'experiment_tag', 'node_ip', 'experiment_id', 'hostname', 'pid', 'date']
    'Results that are saved with `wandb.config` instead of `wandb.log`.'
    _logger_actor_cls = _WandbLoggingActor

    def __init__(self, project: Optional[str]=None, group: Optional[str]=None, api_key_file: Optional[str]=None, api_key: Optional[str]=None, excludes: Optional[List[str]]=None, log_config: bool=False, upload_checkpoints: bool=False, save_checkpoints: bool=False, upload_timeout: int=DEFAULT_SYNC_TIMEOUT, **kwargs):
        if not wandb:
            raise RuntimeError('Wandb was not found - please install with `pip install wandb`')
        if save_checkpoints:
            warnings.warn('`save_checkpoints` is deprecated. Use `upload_checkpoints` instead.', DeprecationWarning)
            upload_checkpoints = save_checkpoints
        self.project = project
        self.group = group
        self.api_key_path = api_key_file
        self.api_key = api_key
        self.excludes = excludes or []
        self.log_config = log_config
        self.upload_checkpoints = upload_checkpoints
        self._upload_timeout = upload_timeout
        self.kwargs = kwargs
        self._remote_logger_class = None
        self._trial_logging_actors: Dict['Trial', ray.actor.ActorHandle[_WandbLoggingActor]] = {}
        self._trial_logging_futures: Dict['Trial', ray.ObjectRef] = {}
        self._logging_future_to_trial: Dict[ray.ObjectRef, 'Trial'] = {}
        self._trial_queues: Dict['Trial', Queue] = {}

    def setup(self, *args, **kwargs):
        self.api_key_file = os.path.expanduser(self.api_key_path) if self.api_key_path else None
        _set_api_key(self.api_key_file, self.api_key)
        self.project = _get_wandb_project(self.project)
        if not self.project:
            raise ValueError(f'Please pass the project name as argument or through the {WANDB_PROJECT_ENV_VAR} environment variable.')
        if not self.group and os.environ.get(WANDB_GROUP_ENV_VAR):
            self.group = os.environ.get(WANDB_GROUP_ENV_VAR)

    def log_trial_start(self, trial: 'Trial'):
        config = trial.config.copy()
        config.pop('callbacks', None)
        exclude_results = self._exclude_results.copy()
        exclude_results += self.excludes
        if not self.log_config:
            exclude_results += ['config']
        trial_id = trial.trial_id if trial else None
        trial_name = str(trial) if trial else None
        wandb_project = self.project
        wandb_group = self.group or trial.experiment_dir_name if trial else None
        config = _clean_log(config)
        config = {key: value for key, value in config.items() if key not in self.excludes}
        wandb_init_kwargs = dict(id=trial_id, name=trial_name, resume=False, reinit=True, allow_val_change=True, group=wandb_group, project=wandb_project, config=config)
        wandb_init_kwargs.update(self.kwargs)
        self._start_logging_actor(trial, exclude_results, **wandb_init_kwargs)

    def _start_logging_actor(self, trial: 'Trial', exclude_results: List[str], **wandb_init_kwargs):
        if not self._remote_logger_class:
            env_vars = {}
            if WANDB_ENV_VAR in os.environ:
                env_vars[WANDB_ENV_VAR] = os.environ[WANDB_ENV_VAR]
            self._remote_logger_class = ray.remote(num_cpus=0, **_force_on_current_node(), runtime_env={'env_vars': env_vars}, max_restarts=-1, max_task_retries=-1)(self._logger_actor_cls)
        self._trial_queues[trial] = Queue(actor_options={'num_cpus': 0, **_force_on_current_node(), 'max_restarts': -1, 'max_task_retries': -1})
        self._trial_logging_actors[trial] = self._remote_logger_class.remote(logdir=trial.local_path, queue=self._trial_queues[trial], exclude=exclude_results, to_config=self.AUTO_CONFIG_KEYS, **wandb_init_kwargs)
        logging_future = self._trial_logging_actors[trial].run.remote()
        self._trial_logging_futures[trial] = logging_future
        self._logging_future_to_trial[logging_future] = trial

    def _signal_logging_actor_stop(self, trial: 'Trial'):
        self._trial_queues[trial].put((_QueueItem.END, None))

    def log_trial_result(self, iteration: int, trial: 'Trial', result: Dict):
        if trial not in self._trial_logging_actors:
            self.log_trial_start(trial)
        result = _clean_log(result)
        self._trial_queues[trial].put((_QueueItem.RESULT, result))

    def log_trial_save(self, trial: 'Trial'):
        if self.upload_checkpoints and trial.checkpoint:
            checkpoint_root = None
            if isinstance(trial.checkpoint.filesystem, pyarrow.fs.LocalFileSystem):
                checkpoint_root = trial.checkpoint.path
            if checkpoint_root:
                self._trial_queues[trial].put((_QueueItem.CHECKPOINT, checkpoint_root))

    def log_trial_end(self, trial: 'Trial', failed: bool=False):
        self._signal_logging_actor_stop(trial=trial)
        self._cleanup_logging_actors()

    def _cleanup_logging_actor(self, trial: 'Trial'):
        del self._trial_queues[trial]
        del self._trial_logging_futures[trial]
        ray.kill(self._trial_logging_actors[trial])
        del self._trial_logging_actors[trial]

    def _cleanup_logging_actors(self, timeout: int=0, kill_on_timeout: bool=False):
        """Clean up logging actors that have finished uploading to wandb.
        Waits for `timeout` seconds to collect finished logging actors.

        Args:
            timeout: The number of seconds to wait. Defaults to 0 to clean up
                any immediate logging actors during the run.
                This is set to a timeout threshold to wait for pending uploads
                on experiment end.
            kill_on_timeout: Whether or not to kill and cleanup the logging actor if
                it hasn't finished within the timeout.
        """
        futures = list(self._trial_logging_futures.values())
        done, remaining = ray.wait(futures, num_returns=len(futures), timeout=timeout)
        for ready_future in done:
            finished_trial = self._logging_future_to_trial.pop(ready_future)
            self._cleanup_logging_actor(finished_trial)
        if kill_on_timeout:
            for remaining_future in remaining:
                trial = self._logging_future_to_trial.pop(remaining_future)
                self._cleanup_logging_actor(trial)

    def on_experiment_end(self, trials: List['Trial'], **info):
        """Wait for the actors to finish their call to `wandb.finish`.
        This includes uploading all logs + artifacts to wandb."""
        self._cleanup_logging_actors(timeout=self._upload_timeout, kill_on_timeout=True)

    def __del__(self):
        if ray.is_initialized():
            for trial in list(self._trial_logging_actors):
                self._signal_logging_actor_stop(trial=trial)
            self._cleanup_logging_actors(timeout=2, kill_on_timeout=True)
        self._trial_logging_actors = {}
        self._trial_logging_futures = {}
        self._logging_future_to_trial = {}
        self._trial_queues = {}