import functools
import logging
import os
import platform
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Type
import ray
from ray.air._internal.session import _get_session
from ray.air._internal.util import RunnerThread, StartTraceback
from ray.air.constants import (
from ray.data import Dataset
from ray.train import Checkpoint
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.storage import StorageContext
from ray.train.constants import (
from ray.train.error import SessionMisuseError
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.placement_group import _valid_resource_shape
from ray.util.scheduling_strategies import (
@DeveloperAPI
class _TrainSession:
    """Holds information for training on each worker."""

    def __init__(self, training_func: Callable, world_rank: int, local_rank: int, node_rank: int, local_world_size: int, world_size: int, trial_info: Optional[TrialInfo]=None, dataset_shard: Optional[Dataset]=None, metadata: Dict[str, Any]=None, checkpoint: Optional[Checkpoint]=None, detailed_autofilled_metrics: bool=False, storage: Optional[StorageContext]=None, synchronous_result_reporting: bool=False):
        self.synchronous_result_reporting = synchronous_result_reporting
        self.dataset_shard = dataset_shard
        self.metadata = metadata
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.local_world_size = local_world_size
        self.world_size = world_size
        assert storage
        logger.debug(f'StorageContext on SESSION (rank={world_rank}):\n{storage}')
        self.reset(training_func=training_func, trial_info=trial_info, storage=storage, loaded_checkpoint=checkpoint)
        self.detailed_autofilled_metrics = detailed_autofilled_metrics
        self.last_report_time = time.time()
        self.iteration = 0
        self.time_total = 0.0
        self.local_ip = self.get_current_ip()
        self.accelerator = None

    def get_current_ip(self):
        self.local_ip = ray.util.get_node_ip_address()
        return self.local_ip

    def start(self):
        """Starts the training thread."""
        self.training_started = True
        self.training_thread.start()

    def reset(self, training_func: Callable, trial_info: TrialInfo, storage: StorageContext, loaded_checkpoint=None):
        self.continue_lock = threading.Semaphore(0)
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue(1)
        self.error_queue = queue.Queue(1)
        self.training_thread = RunnerThread(target=training_func, daemon=True, error_queue=self.error_queue)
        self.trial_info = trial_info
        self.storage = storage
        self.loaded_checkpoint = loaded_checkpoint
        self.ignore_report = False
        self.training_started = False
        self._first_report = True
        os.makedirs(storage.trial_local_path, exist_ok=True)
        if bool(int(os.environ.get(RAY_CHDIR_TO_TRIAL_DIR, '1'))):
            logger.debug(f'Switching the working directory to the trial directory: {storage.trial_local_path}')
            os.chdir(storage.trial_local_path)

    def pause_reporting(self):
        """Ignore all future ``session.report()`` calls."""
        self.ignore_report = True

    def finish(self, timeout: Optional[float]=None):
        """Finishes the training thread.

        Either returns the output from training or raises any Exception from
        training.
        """
        self.stop_event.set()
        self.continue_lock.release()
        self.storage.persist_artifacts(force=True)
        func_output = self.training_thread.join(timeout=timeout)
        return func_output

    def get_next(self) -> Optional[_TrainingResult]:
        """Gets the next ``_TrainingResult`` from the result queue.

        If the result queue is empty, then this function returns ``None``.
        """
        if not self.training_started:
            raise RuntimeError('Please call start before calling get_next.')
        if self.synchronous_result_reporting:
            if not self._first_report:
                self.continue_lock.release()
            self._first_report = False
        result = None
        while result is None and self.training_thread.is_alive():
            try:
                result = self.result_queue.get(block=True, timeout=_RESULT_FETCH_TIMEOUT)
            except queue.Empty:
                pass
        if result is None:
            try:
                result = self.result_queue.get(block=False, timeout=_RESULT_FETCH_TIMEOUT)
            except queue.Empty:
                pass
        if result is None:
            self._report_thread_runner_error(block=True)
        elif not self.error_queue.empty():
            logger.debug('Runner error waiting to be raised in main thread. Logging all available results first.')
        if not self.synchronous_result_reporting:
            self.continue_lock.release()
        return result

    def _auto_fill_metrics(self, result: dict) -> dict:
        """Add autofilled metrics and update attributes."""
        current_time = time.time()
        current_datetime = datetime.now()
        if TIME_THIS_ITER_S in result:
            time_this_iter = result[TIME_THIS_ITER_S]
        else:
            time_this_iter = current_time - self.last_report_time
        self.iteration += 1
        self.time_total += time_this_iter
        self.last_report_time = current_time
        auto_filled_metrics = {TIMESTAMP: int(time.mktime(current_datetime.timetuple())), TIME_TOTAL_S: self.time_total, WORKER_PID: os.getpid(), WORKER_HOSTNAME: platform.node(), WORKER_NODE_IP: self.local_ip}
        if not self.detailed_autofilled_metrics:
            auto_filled_metrics = {k: v for k, v in auto_filled_metrics.items() if k not in DETAILED_AUTOFILLED_KEYS}
        result = result.copy()
        result.update(auto_filled_metrics)
        return result

    def _auto_fill_checkpoint_metrics(self, result: dict) -> dict:
        """Add autofilled metrics and update attributes."""
        current_datetime = datetime.now()
        auto_filled_metrics = {TIMESTAMP: int(time.mktime(current_datetime.timetuple()))}
        result = result.copy()
        result.update(auto_filled_metrics)
        return result

    def _report_thread_runner_error(self, block=False):
        try:
            e = self.error_queue.get(block=block, timeout=_ERROR_FETCH_TIMEOUT)
            raise StartTraceback from e
        except queue.Empty:
            pass

    def _report_training_result(self, training_result: _TrainingResult) -> None:
        """Place a training result on the result queue for the main thread to process,
        then block until the main thread signals that training should continue.

        NOTE: This is used internally to report results from Train to Tune
        without persisting checkpoints to storage 2 times.
        `report` is the public API that directly persists to storage, which
        should only be called by user code.
        """
        if training_result.checkpoint:
            self.loaded_checkpoint = training_result.checkpoint
        self.result_queue.put(training_result, block=True)
        self.continue_lock.acquire()
        if self.stop_event.is_set():
            self.stop_event.clear()
            sys.exit(0)

    def report(self, metrics: Dict, checkpoint: Optional[Checkpoint]=None) -> None:
        if 'torch' in sys.modules:
            from ray.air._internal.torch_utils import contains_tensor
            if contains_tensor(metrics):
                raise ValueError('Passing objects containg Torch tensors as metrics is not supported as it will throw an exception on deserialization. You can either convert the tensors to Python objects or report a `train.Checkpoint` with `ray.train.report` to store your Torch objects.')
        if self.ignore_report:
            return
        metrics = self._auto_fill_metrics(metrics)
        persisted_checkpoint = None
        if checkpoint:
            self.storage._update_checkpoint_index(metrics)
            persisted_checkpoint = self.storage.persist_current_checkpoint(checkpoint)
            metrics[CHECKPOINT_DIR_NAME] = self.storage.checkpoint_dir_name
        else:
            metrics[CHECKPOINT_DIR_NAME] = None
        force_artifact_sync = persisted_checkpoint and self.storage.sync_config.sync_artifacts_on_checkpoint
        self.storage.persist_artifacts(force=force_artifact_sync)
        if persisted_checkpoint and self.metadata:
            user_metadata = persisted_checkpoint.get_metadata()
            for k, v in self.metadata.items():
                if k not in user_metadata:
                    user_metadata[k] = v
            persisted_checkpoint.set_metadata(user_metadata)
        result = _TrainingResult(checkpoint=persisted_checkpoint, metrics=metrics)
        self._report_training_result(result)

    @property
    def experiment_name(self) -> str:
        return self.trial_info.experiment_name

    @property
    def trial_name(self) -> str:
        return self.trial_info.name

    @property
    def trial_id(self) -> str:
        return self.trial_info.id

    @property
    def trial_resources(self) -> 'PlacementGroupFactory':
        return self.trial_info.resources

    @property
    def trial_dir(self) -> str:
        return self.trial_info.logdir

    def get_dataset_shard(self, dataset_name: Optional[str]=None) -> Optional['DataIterator']:
        shard = self.dataset_shard
        if shard is None:
            warnings.warn('No dataset passed in. Returning None. Make sure to pass in a Dataset to Trainer.run to use this function.')
        elif isinstance(shard, dict):
            if not dataset_name:
                raise RuntimeError('Multiple datasets were passed into ``Trainer``, but no ``dataset_name`` is passed into ``get_dataset_shard``. Please specify which dataset shard to retrieve.')
            return shard.get(dataset_name)
        return shard