import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class _StatsManager:
    """A Class containing util functions that manage remote calls to _StatsActor.

    This class collects stats from execution and iteration codepaths and keeps
    track of the latest snapshot.

    An instance of this class runs a single background thread that periodically
    forwards the latest execution/iteration stats to the _StatsActor.

    This thread will terminate itself after being inactive (meaning that there are
    no active executors or iterators) for STATS_ACTOR_UPDATE_THREAD_INACTIVITY_LIMIT
    iterations. After terminating, a new thread will start if more calls are made
    to this class.
    """
    STATS_ACTOR_UPDATE_INTERVAL_SECONDS = 5
    UPDATE_THREAD_INACTIVITY_LIMIT = 5

    def __init__(self):
        self._stats_actor_handle = None
        self._stats_actor_cluster_id = None
        self._last_execution_stats = {}
        self._last_iteration_stats: Dict[str, Tuple[Dict[str, str], 'DatasetStats']] = {}
        self._stats_lock: threading.Lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._update_thread_lock: threading.Lock = threading.Lock()

    def _stats_actor(self, create_if_not_exists=True) -> _StatsActor:
        if ray._private.worker._global_node is None:
            raise RuntimeError('Global node is not initialized.')
        current_cluster_id = ray._private.worker._global_node.cluster_id
        if self._stats_actor_handle is None or self._stats_actor_cluster_id != current_cluster_id:
            if create_if_not_exists:
                self._stats_actor_handle = _get_or_create_stats_actor()
            else:
                self._stat_actor_handle = ray.get_actor(name=STATS_ACTOR_NAME, namespace=STATS_ACTOR_NAMESPACE)
            self._stats_actor_cluster_id = current_cluster_id
        return self._stats_actor_handle

    def _start_thread_if_not_running(self):
        with self._update_thread_lock:
            if self._update_thread is None or not self._update_thread.is_alive():

                def _run_update_loop():
                    iter_stats_inactivity = 0
                    while True:
                        if self._last_iteration_stats or self._last_execution_stats:
                            try:
                                self._stats_actor(create_if_not_exists=False).update_metrics.remote(execution_metrics=list(self._last_execution_stats.values()), iteration_metrics=list(self._last_iteration_stats.values()))
                                iter_stats_inactivity = 0
                            except Exception:
                                logger.get_logger(log_to_stdout=False).exception('Error occurred during remote call to _StatsActor.')
                                return
                        else:
                            iter_stats_inactivity += 1
                            if iter_stats_inactivity >= _StatsManager.UPDATE_THREAD_INACTIVITY_LIMIT:
                                logger.get_logger(log_to_stdout=False).info('Terminating StatsManager thread due to inactivity.')
                                return
                        time.sleep(StatsManager.STATS_ACTOR_UPDATE_INTERVAL_SECONDS)
                self._update_thread = threading.Thread(target=_run_update_loop, daemon=True)
                self._update_thread.start()

    def update_execution_metrics(self, dataset_tag: str, op_metrics: List[OpRuntimeMetrics], operator_tags: List[str], state: Dict[str, Any], force_update: bool=False):
        op_metrics_dicts = [metric.as_dict() for metric in op_metrics]
        args = (dataset_tag, op_metrics_dicts, operator_tags, state)
        if force_update:
            self._stats_actor().update_execution_metrics.remote(*args)
        else:
            with self._stats_lock:
                self._last_execution_stats[dataset_tag] = args
            self._start_thread_if_not_running()

    def clear_execution_metrics(self, dataset_tag: str, operator_tags: List[str]):
        with self._stats_lock:
            if dataset_tag in self._last_execution_stats:
                del self._last_execution_stats[dataset_tag]
        try:
            self._stats_actor(create_if_not_exists=False).clear_execution_metrics.remote(dataset_tag, operator_tags)
        except Exception:
            pass

    def update_iteration_metrics(self, stats: 'DatasetStats', dataset_tag: str):
        with self._stats_lock:
            self._last_iteration_stats[dataset_tag] = (stats, dataset_tag)
        self._start_thread_if_not_running()

    def clear_iteration_metrics(self, dataset_tag: str):
        with self._stats_lock:
            if dataset_tag in self._last_iteration_stats:
                del self._last_iteration_stats[dataset_tag]
        try:
            self._stats_actor(create_if_not_exists=False).clear_iteration_metrics.remote(dataset_tag)
        except Exception:
            pass

    def register_dataset_to_stats_actor(self, dataset_tag, operator_tags):
        self._stats_actor().register_dataset.remote(dataset_tag, operator_tags)

    def get_dataset_id_from_stats_actor(self) -> str:
        try:
            return ray.get(self._stats_actor().get_dataset_id.remote())
        except Exception:
            return uuid4().hex