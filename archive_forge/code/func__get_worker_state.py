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
def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
    failure = result.failures.get(worker.global_rank)
    if result.state in {WorkerState.UNHEALTHY, WorkerState.FAILED} and (not failure):
        return 'TERMINATED'
    elif failure or worker.global_rank in result.return_values:
        return result.state.value
    else:
        raise ValueError(f'Unknown worker: {worker.global_rank}')