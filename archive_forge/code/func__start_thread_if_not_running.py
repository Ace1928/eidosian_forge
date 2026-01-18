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