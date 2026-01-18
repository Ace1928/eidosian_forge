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
def _record_worker_events(self, result: RunResult) -> None:
    for worker in self._worker_group.workers:
        failure = result.failures.get(worker.global_rank)
        state: str = self._get_worker_state(worker, result)
        raw_error = json.dumps(failure.error_file_data) if failure else None
        record(self._construct_event(state, EventSource.WORKER, worker, raw_error))