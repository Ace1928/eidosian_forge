import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
def _exit_barrier(self):
    """
        Define a barrier that keeps the agent process alive until all workers finish.

        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times.
        """
    log.info('Local worker group finished (%s). Waiting %s seconds for other agents to finish', self._worker_group.state, self._exit_barrier_timeout)
    start = time.time()
    try:
        store_util.barrier(self._store, self._worker_group.group_rank, self._worker_group.group_world_size, key_prefix=_TERMINAL_STATE_SYNC_ID, barrier_timeout=self._exit_barrier_timeout)
        log.info('Done waiting for other agents. Elapsed: %s seconds', time.time() - start)
    except SignalException as e:
        log.warning('Got termination signal: %s', e.sigval)
        raise
    except Exception:
        log.exception('Error waiting on exit barrier. Elapsed: %s seconds', time.time() - start)